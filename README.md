Real-ESRGAN Image Upscaler

This script utilizes Real-ESRGAN to upscale images, with options for denoising and specifying the output directory and upscaling factor. It is designed for use with document images but can be adapted for other types of images.
Features:

    Upscale images by a factor of 2, 4, or other values (via --outscale).
    Control noise reduction strength with --denoise_strength (from 0 to 1).
    Specify input and output directories.
    Save results in a custom output folder.

Requirements:

    Python 3.x
    OpenCV for image processing:

    pip install opencv-python

    Real-ESRGAN and other dependencies. Follow the instructions in the Real-ESRGAN repository to install the required files and weights:
        Real-ESRGAN GitHub Repository

Setup:

    Clone the repository:

git clone https://github.com/ahrbilzakaria/Document_Text_Enhancing.git
cd Document_Text_Enhancing

Install dependencies:

    pip install -r requirements.txt

    Download the model weights: Ensure you have downloaded the model weights (realesr-general-x4v3.pth) into the weights folder. If not already present, the script will attempt to download them automatically.

Usage:

To use the script, run the following command from the terminal:

python upscale.py <input_folder> --output_dir <output_folder> --outscale <upscale_factor> --denoise_strength <noise_strength>

Arguments:

    input_folder (required): The folder containing the images to be upscaled.
    --output_dir (optional): The folder where upscaled images will be saved (default is results).
    --outscale (optional): The scale factor for upscaling the images (default is 4).
    --denoise_strength (optional): Strength of noise reduction, from 0 (no denoising) to 1 (maximum denoising, default is 0.5).

Example:

Upscale images in the documents folder by a factor of 2 and apply a denoise strength of 0.7. The results will be saved in the output folder:

python upscale.py documents --output_dir output --outscale 2 --denoise_strength 0.7

Output:

The upscaled images will be saved in the specified output_dir folder with the suffix _upscaled added to their original names.
