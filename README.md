# Real-ESRGAN Image Upscaler

This script uses **Real-ESRGAN** to upscale images, with options for denoising, specifying the output directory, and setting the upscaling factor. It is designed for document images but can be adapted for other types of images.

---

## Features

- Upscale images by a factor of 2, 4, or other values (via `--outscale`).
- Control noise reduction strength with `--denoise_strength` (from 0 to 1).
- Specify input and output directories.
- Save results in a custom output folder.

---

## Requirements

- **Python 3.x**
- **OpenCV** for image processing:
  ```bash
  pip install opencv-python
  ```
- **Real-ESRGAN** and other dependencies. Follow the instructions in the [Real-ESRGAN GitHub Repository](https://github.com/xinntao/Real-ESRGAN) to install the required files and weights.

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ahrbilzakaria/Document_Text_Enhancing.git
   cd Document_Text_Enhancing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model weights:
   - Ensure you have downloaded the model weights (`realesr-general-x4v3.pth`) into the `weights` folder.
   - If the weights are not already present, the script will attempt to download them automatically.

---

## Usage

Run the script from the terminal using the following command:

```bash
python upscale.py <input_folder> --output_dir <output_folder> --outscale <upscale_factor> --denoise_strength <noise_strength>
```

### Arguments

- `input_folder` (required): The folder containing the images to be upscaled.
- `--output_dir` (optional): The folder where upscaled images will be saved (default is `results`).
- `--outscale` (optional): The scale factor for upscaling the images (default is `4`).
- `--denoise_strength` (optional): Strength of noise reduction, from `0` (no denoising) to `1` (maximum denoising, default is `0.5`).

### Example

Upscale images in the `documents` folder by a factor of 2 and apply a denoise strength of 0.7. The results will be saved in the `output` folder:

```bash
python upscale.py documents --output_dir output --outscale 2 --denoise_strength 0.7
```

---

## Output

The upscaled images will be saved in the specified `output_dir` folder with the suffix `_upscaled` added to their original names.

---

## Notes

- Ensure the input images are in a supported format (e.g., `.jpg`, `.png`).
- For best results, use high-quality input images.
- Adjust the `--denoise_strength` parameter based on the noise level in your images.

---

## Acknowledgments

- Real-ESRGAN: [GitHub Repository](https://github.com/xinntao/Real-ESRGAN)
- OpenCV: [OpenCV Documentation](https://docs.opencv.org/)
