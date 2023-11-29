import sys
sys.path.append(".")
sys.path.append("..")
import torch
import rawpy
import torch
import numpy as np
from PIL import Image
import numpy as np
from collections import OrderedDict as odict
import imageio.v2 as imageio

from util.utils_single_image import extract_bayer_channels, meta_data_process, get_attention_coords
from options.test_options import TestOptions
from options.single_options import SingleOptions
from models import create_model


def main(args, devices_dict):
    """
    Run inference on a single image using the given arguments

    Args:
        args (argparse.Namespace): Command-line arguments.
        devices_dict (dict): Dictionary mapping device indices to device names.

    Returns:
        None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current Device: ", device)
    model_load = torch.load(args.path_model)
    model = create_model(args)
    model.setup(args)

    msg = model.netMetaISPNet.load_state_dict(model_load, strict=True)
    print(msg)
    
    print("Reading and pre-processing metadata")
    color_scaler, exp, iso = meta_data_process(args.path_image, args, device)
    image_size = args.img_size

    ## READ RAW
    print("Loading RAW image")
    raw = rawpy.imread(args.path_image)
    raw_input = raw.raw_image
    raw_input = raw_input.astype(np.uint16) 

    # READ RGB
    try:
        rgb = imageio.imread(args.path_image[:-4]+'.jpg')
    except:
        rgb = imageio.imread(args.path_image[:-4]+'.JPG')
    
    # The images are cropped to have the same size as the ones used on paper. This makes the images squared over the downsampling process. 
    original_h, original_w = raw_input.shape
    crop_h = (original_h - image_size) // 2
    crop_w = (original_w - image_size) // 2
    raw_input = raw_input[crop_h:-(crop_h), crop_w:-crop_w]
    rgb = rgb[crop_h:-(crop_h), crop_w:-crop_w, :]

    # Extract bayer channels and get downsampled bilinear interpolated image
    raw, bilinear = extract_bayer_channels(raw_input, args)
    raw = torch.unsqueeze(torch.from_numpy(raw.transpose((2, 0, 1))), 0).to(device)
    bilinear = torch.unsqueeze(torch.from_numpy(bilinear.transpose((2, 0, 1))), 0).to(device)

    # Get attention coordinates
    coords_atten = get_attention_coords(args)
    
    with torch.no_grad():
        model.eval()
        print("Running inference...")
        for i in range(args.pre_trained_devices):
            data = {
                'raw': raw,
                'dslr': bilinear,
                'fname': 'test',
                'raw_demosaic_full': bilinear,
                'wb': color_scaler,
                'dslr_image_ref': bilinear,
                'device': torch.tensor([i]),
                'coords': coords_atten,
                'exp': exp,
                'iso': iso
            }

            model.set_input(data)
            model.test()
            res = model.get_current_visuals()
            image = np.array((res['data_out'][0]).permute(1, 2, 0).cpu()).astype(np.uint8)
            im = Image.fromarray(np.uint8(image))
            im.save('images_test/test_' + devices_dict[i] + '.png') 

        rgb = Image.fromarray(np.uint8(rgb))
        rgb.save('images_test/original_noisy' + str(i) + '.png') 

        try:
            rgb_2 = imageio.imread(args.path_image[:-4] + '_p.jpg')
            rgb_2 = rgb_2[crop_h:-(crop_h), crop_w:-crop_w, :]
            rgb_2 = Image.fromarray(np.uint8(rgb_2))
            rgb_2.save('images_test/original' + str(i) + '.png') 
        except:
            pass

        print("Done!")


if __name__ == '__main__':
    
    args = SingleOptions().parse()
    devices_dict = {0:'pixel', 1:'samsung', 2:'iphone'}

    main(args,devices_dict)
 