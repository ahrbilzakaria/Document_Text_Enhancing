
# 📄Document Text Enhancing Pipeline

This project enhances document images using a pipeline of image processing techniques: **Real-ESRGAN** for upscaling, **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement, and **adaptive unsharp masking** for sharpening.

---

## Features

- **Upscale** images using Real-ESRGAN by a factor of 2, 4, or other values (via `--outscale`).
- **Enhance contrast** using CLAHE for improved readability of documents.
- **Apply adaptive sharpening** to fine-tune image details after upscaling and contrast enhancement.
- Control noise reduction strength with `--denoise_strength` (from 0 to 1).
- Specify input and output directories.
- Save results in dedicated output folders for each processing step.

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
   git clone https://github.com/ahrbilzakaria/Document_Text_Enhancing_For_Ocr.git
   cd Document_Text_Enhancing_For_Ocr
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

Run the pipeline script from the terminal to process images through the upscaling, contrast enhancement, and sharpening steps:

```bash
python pipeline.py
```

### Pipeline Steps

1. **Upscaling**:
   - Images are upscaled using the **Real-ESRGAN** model by the specified scale factor (default is `4`).
   
2. **Contrast Enhancement**:
   - **CLAHE** (Contrast Limited Adaptive Histogram Equalization) is applied to enhance the contrast of the upscaled images, improving their readability.

3. **Sharpening**:
   - **Adaptive Unsharp Masking** is used to sharpen the processed images, bringing out finer details.

### Input and Output Folders

- **Input Folder** (`start`): Contains the original images to be processed.
- **Output Folders**:
  - **`output/`**: Contains the upscaled images.
  - **`clahe_output/`**: Contains the images with enhanced contrast using CLAHE.
  - **`sharpen_output/`**: Contains the final sharpened images after applying unsharp masking.

---

## Example

To run the pipeline with default settings:

```bash
python pipeline.py
```

This will:
1. Upscale images from the `start/` folder by a factor of 2 (default scale).
2. Enhance contrast using CLAHE.
3. Apply adaptive sharpening.
4. Save the processed images in `output/`, `clahe_output/`, and `sharpen_output/` folders.

You can modify the pipeline code to change input/output directories or any processing parameters.

---

## Notes

- Ensure the input images are in a supported format (e.g., `.jpg`, `.png`).
- For best results, use high-quality input images.
- The script supports adjusting noise reduction (`--denoise_strength`) and upscaling factor (`--outscale`) during the upscaling step.
- The output images will be saved with appropriate suffixes (`_upscaled`, `_clahe`, `_sharpened`) in their respective directories.

---

## Acknowledgments

- **Real-ESRGAN**: [GitHub Repository](https://github.com/xinntao/Real-ESRGAN)
- **OpenCV**: [OpenCV Documentation](https://docs.opencv.org/)
- **CLAHE** and **Adaptive Unsharp Masking**: Custom implementations for contrast enhancement and sharpening.
