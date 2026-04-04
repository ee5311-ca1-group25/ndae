#!/usr/bin/env python3
"""Download a tiny local SVBRDF subset from the official NDAE dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

from remotezip import RemoteZip


DATASET_PAGE_URL = (
    "https://rdr.ucl.ac.uk/articles/dataset/"
    "Dynamic_Flash_Video_Data_Ready-to-use_with_Time-varying_Appearance/"
    "27284520?file=49943397"
)
DATASET_DOI_URL = "https://doi.org/10.5522/04/27284520"
DATASET_DOWNLOAD_URL = "https://rdr.ucl.ac.uk/ndownloader/files/49943397"
PWCLI_BASE_CMD = ["npx", "--yes", "--package", "@playwright/cli", "playwright-cli"]
COOKIE_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9_.-]+)=(?P<value>[^ ]+) "
    r"\(domain: (?P<domain>[^,]+), path: (?P<path>[^)]+)\)"
)


class DownloadError(RuntimeError):
    """Raised when the SVBRDF mini download flow fails."""


@dataclass(slots=True)
class DownloadManifest:
    exemplar: str
    selected_files: list[str]
    output_dir: str
    source_page_url: str
    source_download_url: str
    generated_at_utc: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a tiny SVBRDF subset into data_local/svbrdf_mini.",
    )
    parser.add_argument(
        "--exemplar",
        default="clay_solidifying",
        help="Exemplar folder to sample from inside the official ZIP.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of JPG frames to sample uniformly when --files is not provided.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional explicit JPG basenames or archive paths to download.",
    )
    parser.add_argument(
        "--output-root",
        default="data_local/svbrdf_mini",
        help="Root folder for the local mini dataset.",
    )
    parser.add_argument(
        "--semi-auto",
        action="store_true",
        help=(
            "Open the dataset page in your regular browser and prompt for a "
            "Cookie header or signed URL copied from DevTools."
        ),
    )
    parser.add_argument(
        "--cookie-header",
        default=None,
        help=(
            "Optional cookie header for rdr.ucl.ac.uk, e.g. "
            "'aws-waf-token=...; FIGINSTWEBIDCD=...'. When provided, "
            "the script skips Playwright cookie acquisition."
        ),
    )
    parser.add_argument(
        "--signed-url",
        default=None,
        help=(
            "Optional pre-signed S3 ZIP URL copied from a successful browser "
            "download redirect. When provided, the script skips cookie "
            "acquisition and signed URL minting."
        ),
    )
    parser.add_argument(
        "--page-url",
        default=DATASET_DOI_URL,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--download-url",
        default=DATASET_DOWNLOAD_URL,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--session-name",
        default=f"svm-{int(time.time())}",
        help="Playwright CLI session name used to acquire the WAF cookie.",
    )
    parser.add_argument(
        "--list-exemplars",
        action="store_true",
        help="Print available exemplar folders and exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite local files if they already exist.",
    )
    return parser


def run_pwcli(session_name: str, *args: str, timeout: int = 120) -> str:
    cmd = [*PWCLI_BASE_CMD, f"-s={session_name}", *args]
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise DownloadError(f"Playwright CLI failed for `{ ' '.join(args) }`: {details}")
    return result.stdout


def normalize_session_name(raw_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", raw_name).strip("-_")
    if not cleaned:
        cleaned = "svm"
    if len(cleaned) <= 12:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:6]
    prefix = cleaned[:5].rstrip("-_") or "svm"
    return f"{prefix}-{digest}"


def ensure_prerequisites() -> None:
    if shutil.which("npx") is None:
        raise DownloadError("`npx` is required. Install Node.js/npm first.")


def open_in_system_browser(url: str) -> None:
    opened = webbrowser.open(url, new=1)
    if not opened:
        print(f"Open this URL in your browser: {url}")


def open_dataset_page(session_name: str, page_url: str) -> None:
    run_pwcli(session_name, "open", page_url, timeout=180)


def close_dataset_page(session_name: str) -> None:
    try:
        run_pwcli(session_name, "close", timeout=30)
    except Exception:
        pass


def get_page_title(session_name: str) -> str:
    output = run_pwcli(session_name, "eval", "document.title", timeout=30)
    match = re.search(r'^### Result\s+"(?P<title>.*)"$', output, re.MULTILINE)
    return match.group("title") if match else ""


def wait_for_cookie_header(session_name: str, timeout_s: int = 30) -> str:
    deadline = time.time() + timeout_s
    saw_403 = False
    while time.time() < deadline:
        cookies = parse_rdr_cookies(run_pwcli(session_name, "cookie-list", timeout=30))
        if any(name == "aws-waf-token" for name, _, _ in cookies):
            return "; ".join(f"{name}={value}" for name, value, _ in cookies)
        title = get_page_title(session_name)
        if title == "403 Forbidden":
            saw_403 = True
            break
        time.sleep(1.0)
    if saw_403:
        raise DownloadError(
            "The UCL dataset site returned `403 Forbidden` to the automated browser. "
            "Automatic cookie acquisition is blocked right now. "
            "Retry later, or rerun the script with `--semi-auto`, "
            "`--cookie-header`, or `--signed-url`."
        )
    raise DownloadError(
        "Timed out waiting for the `aws-waf-token` browser cookie. "
        "Retry later, or rerun the script with `--semi-auto`, "
        "`--cookie-header`, or `--signed-url`."
    )


def parse_rdr_cookies(cookie_output: str) -> list[tuple[str, str, str]]:
    cookies: list[tuple[str, str, str]] = []
    for line in cookie_output.splitlines():
        match = COOKIE_PATTERN.search(line.strip())
        if not match:
            continue
        domain = match.group("domain")
        if "rdr.ucl.ac.uk" not in domain:
            continue
        cookies.append((match.group("name"), match.group("value"), domain))
    return cookies


def normalize_cookie_header(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise DownloadError("Cookie header input is empty.")

    if value.lower().startswith("cookie:"):
        value = value.split(":", 1)[1].strip()
    else:
        match = re.search(r"cookie:\s*(.+)", value, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1).strip()

    value = value.replace("\r", "\n")
    parts: list[str] = []
    for segment in re.split(r"[;\n]+", value):
        segment = segment.strip()
        if not segment or "=" not in segment:
            continue
        name, cookie_value = segment.split("=", 1)
        name = name.strip()
        cookie_value = cookie_value.strip()
        if not name or not cookie_value:
            continue
        parts.append(f"{name}={cookie_value}")

    if not parts:
        raise DownloadError(
            "Could not parse a usable Cookie header. Paste the request's "
            "`Cookie:` header value or the full `Cookie: ...` line."
        )
    return "; ".join(parts)


def parse_manual_input(raw: str) -> tuple[str | None, str | None]:
    value = raw.strip()
    if not value:
        raise DownloadError("No manual input was provided.")
    if value.startswith("https://"):
        return None, value
    return normalize_cookie_header(value), None


def prompt_for_manual_access(page_url: str, download_url: str) -> tuple[str | None, str | None]:
    if not sys.stdin.isatty():
        raise DownloadError(
            "Semi-automatic mode requires an interactive terminal. "
            "Use `--cookie-header` or `--signed-url` in non-interactive runs."
        )

    instructions = dedent(
        f"""
        Semi-automatic download flow:

        1. Your default browser will open the UCL dataset page.
        2. In that browser, make sure the page loads normally.
        3. Open DevTools -> Network.
        4. Click the dataset Download button once.
        5. Prefer copying the `Cookie` request header from:
           `{download_url}`
           You can also copy the redirected S3 ZIP URL instead.
        6. Paste that Cookie header or signed URL back here.
        """
    ).strip()
    print(instructions)
    open_in_system_browser(page_url)

    while True:
        try:
            pasted = input("\nPaste Cookie header or signed URL: ").strip()
        except EOFError as exc:
            raise DownloadError("Input stream closed before manual access data was provided.") from exc

        if not pasted:
            print("Input was empty. Paste a Cookie header or signed URL.")
            continue
        try:
            return parse_manual_input(pasted)
        except DownloadError as exc:
            print(f"Invalid input: {exc}")
            continue


def mint_signed_url(cookie_header: str, download_url: str, referer: str) -> str:
    request = urllib.request.Request(
        download_url,
        headers={
            "Cookie": cookie_header,
            "Range": "bytes=0-1023",
            "Referer": referer,
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206:
            raise DownloadError(
                f"Expected HTTP 206 while minting the signed URL, got {response.status}."
            )
        return response.geturl()


def resolve_signed_url(
    signed_url: str | None,
    cookie_header: str | None,
    download_url: str,
    referer: str,
) -> str:
    if signed_url:
        return signed_url
    if cookie_header:
        return mint_signed_url(cookie_header, download_url, referer)
    raise DownloadError("Either `signed_url` or `cookie_header` must be available.")


def list_archive_jpgs(signed_url: str, exemplar: str | None = None) -> list[str]:
    with RemoteZip(signed_url) as archive:
        names = archive.namelist()
    jpgs = [name for name in names if name.lower().endswith(".jpg")]
    if exemplar is None:
        return jpgs
    prefix = exemplar.strip("/") + "/"
    return sorted(name for name in jpgs if name.startswith(prefix))


def list_exemplars(signed_url: str) -> list[str]:
    return sorted({name.split("/", 1)[0] for name in list_archive_jpgs(signed_url)})


def select_uniform_files(files: list[str], count: int) -> list[str]:
    if count <= 0:
        raise DownloadError("--count must be greater than 0.")
    if count >= len(files):
        return files
    if count == 1:
        return [files[0]]

    indices: list[int] = []
    for position in range(count):
        index = round(position * (len(files) - 1) / (count - 1))
        if index not in indices:
            indices.append(index)

    if len(indices) < count:
        for index in range(len(files)):
            if index in indices:
                continue
            indices.append(index)
            if len(indices) == count:
                break
        indices.sort()

    return [files[index] for index in indices]


def resolve_explicit_files(files: list[str], requested: list[str]) -> list[str]:
    by_name = {Path(name).name: name for name in files}
    resolved: list[str] = []
    for item in requested:
        if item in files:
            resolved.append(item)
            continue
        if item in by_name:
            resolved.append(by_name[item])
            continue
        raise DownloadError(f"Could not find `{item}` under the selected exemplar.")
    return resolved


def download_selected_files(
    selected_files: list[str],
    output_dir: Path,
    cookie_header: str | None,
    download_url: str,
    referer: str,
    overwrite: bool,
    signed_url: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for archive_path in selected_files:
        target = output_dir / Path(archive_path).name
        if target.exists() and not overwrite:
            print(f"skip {target} already exists ({target.stat().st_size} bytes)")
            continue

        current_signed_url = resolve_signed_url(
            signed_url=signed_url,
            cookie_header=cookie_header,
            download_url=download_url,
            referer=referer,
        )
        with RemoteZip(current_signed_url) as archive:
            data = archive.read(archive_path)
        target.write_bytes(data)
        print(f"wrote {target} {len(data)}")


def write_manifest(
    exemplar: str,
    selected_files: list[str],
    output_dir: Path,
    page_url: str,
    download_url: str,
) -> None:
    manifest = DownloadManifest(
        exemplar=exemplar,
        selected_files=selected_files,
        output_dir=str(output_dir),
        source_page_url=page_url,
        source_download_url=download_url,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path = output_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    ensure_prerequisites()

    session_name = normalize_session_name(args.session_name)
    cookie_header = args.cookie_header
    try:
        if args.signed_url:
            signed_url = args.signed_url
        else:
            if cookie_header is None and args.semi_auto:
                cookie_header, signed_url = prompt_for_manual_access(
                    DATASET_PAGE_URL,
                    args.download_url,
                )
            else:
                signed_url = None

            if cookie_header is None and signed_url is None:
                open_dataset_page(session_name, args.page_url)
                cookie_header = wait_for_cookie_header(session_name)
            if signed_url is None:
                signed_url = mint_signed_url(cookie_header, args.download_url, DATASET_PAGE_URL)

        if args.list_exemplars:
            for exemplar in list_exemplars(signed_url):
                print(exemplar)
            return 0

        exemplar_files = list_archive_jpgs(signed_url, args.exemplar)
        if not exemplar_files:
            raise DownloadError(f"No JPG files found for exemplar `{args.exemplar}`.")

        if args.files:
            selected_files = resolve_explicit_files(exemplar_files, args.files)
        else:
            selected_files = select_uniform_files(exemplar_files, args.count)

        output_dir = Path(args.output_root) / args.exemplar
        download_selected_files(
            selected_files,
            output_dir,
            cookie_header,
            args.download_url,
            DATASET_PAGE_URL,
            args.overwrite,
            signed_url=args.signed_url,
        )
        write_manifest(
            args.exemplar,
            selected_files,
            output_dir,
            args.page_url,
            args.download_url,
        )
        return 0
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        close_dataset_page(session_name)


if __name__ == "__main__":
    raise SystemExit(main())
