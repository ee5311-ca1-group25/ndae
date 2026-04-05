# Neural ODE Texture Generation

A deep learning project that trains a UNet-based Neural ODE model to generate and animate textures from a GIF using neural style transfer.

## How It Works
- Loads frames from an input GIF as style targets
- Uses a UNet with circular convolutions (for tileable textures) and sinusoidal time embeddings
- Integrates the UNet as a Neural ODE using `torchdiffeq`
- Optimizes using VGG16-based perceptual style loss (Gram matrix + Sliced Wasserstein Distance)
- Outputs synthesized animated frames that match the style of the input GIF

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your input GIF
Place your GIF inside the `data/` folder and update the `gif_path` in the `Config` class in `main_rgb.py`:
```python
class Config:
    gif_path = "data/your_file.gif"
```

### 3. Run
```bash
python main_rgb.py
```

## Requirements
- Python 3.8+
- CUDA-capable GPU recommended (will fall back to CPU automatically)

## Project Structure
```
├── main_rgb.py        # Main script
├── data/              # Put your input GIF here
├── requirements.txt   # Python dependencies
├── .gitignore         # Files excluded from version control
└── README.md          # This file
```
