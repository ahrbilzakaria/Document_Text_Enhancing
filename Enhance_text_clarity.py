import cv2
import glob
import os
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url

def upscale_images(input_dir='inputs', model_name='realesr-general-x4v3', output_dir='results', denoise_strength=0.5, outscale=4, model_path=None, suffix='upscaled', tile=0, tile_pad=10, pre_pad=0, ext='auto', gpu_id=None):
    """ Upscales images using Real-ESRGAN. """

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
    if model_path is not None:
        model_path = model_path
    else:
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
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        gpu_id=gpu_id)

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
            if ext == 'auto':
                extension = extension[1:]
            else:
                extension = ext
            if img_mode == 'RGBA':
                extension = 'png'
            if suffix == '':
                save_path = os.path.join(output_dir, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(output_dir, f'{imgname}_{suffix}.{extension}')
            cv2.imwrite(save_path, output)

# Example of calling the function:
upscale_images(input_dir='start', output_dir='output', outscale=2, denoise_strength=0.5)
