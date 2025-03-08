import argparse
import cv2
import glob
import os
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url

def upscale_images(input_dir='inputs', output_dir='results', outscale=4, denoise_strength=0.5, model_name='realesr-general-x4v3'):
    """ Upscales images using Real-ESRGAN with optional denoise strength. """

    # Set model parameters
    model_name = model_name.split('.')[0]
    if model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # Determine model paths
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # Use dni to control the denoise strength
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # Initialize the upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model)

    os.makedirs(output_dir, exist_ok=True)

    # Get input images
    if os.path.isfile(input_dir):
        paths = [input_dir]
    else:
        paths = sorted(glob.glob(os.path.join(input_dir, '*')))

    # Process each image
    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if extension == 'auto':
                extension = extension[1:]
            else:
                extension = extension
            if img_mode == 'RGBA':
                extension = 'png'
            save_path = os.path.join(output_dir, f'{imgname}_upscaled{extension}')
            cv2.imwrite(save_path, output)

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Upscale images using Real-ESRGAN")
    parser.add_argument('input_dir', type=str, help="Directory containing images to upscale")
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save upscaled images")
    parser.add_argument('--outscale', type=int, default=4, help="Scale factor for upscaling")
    parser.add_argument('--denoise_strength', type=float, default=0.5, help="Strength of denoising (0 to 1)")

    # Parse the arguments
    args = parser.parse_args()

    # Call the upscale_images function with parsed arguments
    upscale_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        outscale=args.outscale,
        denoise_strength=args.denoise_strength
    )
