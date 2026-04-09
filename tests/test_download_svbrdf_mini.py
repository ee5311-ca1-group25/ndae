from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_svbrdf_mini.py"
SPEC = spec_from_file_location("download_svbrdf_mini", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_normalize_cookie_header_accepts_full_cookie_line() -> None:
    value = MODULE.normalize_cookie_header(
        "Cookie: aws-waf-token=token123; FIGINSTWEBIDCD=session456"
    )
    assert value == "aws-waf-token=token123; FIGINSTWEBIDCD=session456"


def test_normalize_cookie_header_accepts_multiline_headers() -> None:
    value = MODULE.normalize_cookie_header(
        "Cookie: aws-waf-token=token123\r\nFIGINSTWEBIDCD=session456\r\nfoo=bar"
    )
    assert value == "aws-waf-token=token123; FIGINSTWEBIDCD=session456; foo=bar"


def test_parse_manual_input_accepts_signed_url() -> None:
    cookie_header, signed_url = MODULE.parse_manual_input(
        "https://s3-eu-west-1.amazonaws.com/example.zip?sig=abc"
    )
    assert cookie_header is None
    assert signed_url == "https://s3-eu-west-1.amazonaws.com/example.zip?sig=abc"


def test_parse_manual_input_rejects_empty_value() -> None:
    with pytest.raises(MODULE.DownloadError, match="No manual input"):
        MODULE.parse_manual_input("   ")
