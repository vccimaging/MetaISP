import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np
import cv2
import torch
import exifread
import math


def extract_bayer_channels(raw, args):
    """
    Extracts the Bayer channels from a raw image and performs bilinear interpolation.

    Args:
        raw (ndarray): Input raw image.
        args (Namespace): Command-line arguments.

    Returns:
        tuple: A tuple containing two images:
            - The normalized raw image with black level removed.
            - The bilinear interpolated image.
    """

    # Reshape the input bayer image
    if args.device == 'xiaomi':
        ch_B  = raw[1::2, 0::2]
        ch_Gb = raw[0::2, 0::2]
        ch_R  = raw[0::2, 1::2]
        ch_Gr = raw[1::2, 1::2]
    elif args.device == 'iphone':
        ch_R  = raw[0::2, 0::2]
        ch_Gb = raw[0::2, 1::2]
        ch_Gr = raw[1::2, 0::2]
        ch_B  = raw[1::2, 1::2]
    else:
        raise ValueError('Device not supported')

    # Combine the channels
    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32)

    # Bilinear interpolation
    ch_B = cv2.resize(ch_B, (ch_B.shape[1] * 2, ch_B.shape[0] * 2))
    ch_R = cv2.resize(ch_R, (ch_R.shape[1] * 2, ch_R.shape[0] * 2))
    ch_Gb = cv2.resize(ch_Gb, (ch_Gb.shape[1] * 2, ch_Gb.shape[0] * 2))
    ch_Gr = cv2.resize(ch_Gr, (ch_Gr.shape[1] * 2, ch_Gr.shape[0] * 2))
    ch_G = ch_Gb / 2 + ch_Gr / 2
    bilinear = np.dstack((ch_B, ch_G, ch_R))
    bilinear = bilinear.astype(np.uint16) 
    bilinear = cv2.resize(bilinear,(args.bilinear_size,args.bilinear_size), interpolation = cv2.INTER_AREA)

    return remove_black_level(RAW_norm,args),remove_black_level(bilinear,args)

def remove_black_level(img, args):
    """
    Removes the black level from the input image based on the device type.

    Args:
        img (numpy.ndarray): The input image.
        args (argparse.Namespace): The command line arguments.

    Returns:
        numpy.ndarray: The image with the black level removed.
    """
    if args.device == 'xiaomi':
        black_lv = 64
        white_lv = 1023
    elif args.device == 'iphone':
        black_lv = 528
        white_lv = 4095
    else:
        raise ValueError('Device not supported')
    img = np.maximum(img.astype(np.float32) - black_lv, 0) / (white_lv - black_lv)
    return img

def meta_data_process(raw_file, args, device):
    """
    Process metadata from an image file.

    Args:
        raw_file (str): The path to the raw image file.
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: A tuple containing the processed color scaler, exposure time, and ISO speed ratings pre-processed.
    """
    # Read metadata using exifread library
    f = open(raw_file, "rb")
    tags = exifread.process_file(f)
    wb = tags["Image Tag 0xC628"]

    if args.device == 'xiaomi':
        if args.iso_exp:
            exp = torch.clip(torch.tensor(float(tags["Image ExposureTime"].values[0])).unsqueeze(0)*10,0,6)
            iso = torch.clip(torch.tensor(float(tags["Image ISOSpeedRatings"].values[0])).unsqueeze(0)/100,0,1)

    elif args.device == 'iphone':
        exp = torch.tensor(float(tags["EXIF ExposureTime"].values[0]))
        exp = (exp.unsqueeze(0)*100)
        iso = torch.tensor(float(tags["EXIF ISOSpeedRatings"].values[0]))
        iso = (iso.unsqueeze(0)/100)
    else:
        raise ValueError('Device not supported')
    
    r1, r2 = str(wb.values[0]).split("/",2)
    r1 = int(r1)
    r2 = int(r2)
    r_ = (1/(r1/r2))
    b1, b2 = str(wb.values[2]).split("/",2)
    b1 = int(b1)
    b2 = int(b2)
    b_ = (1/(b1/b2))
    g = int(wb.values[1])
    
    r_meta = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor(r_),0),1),1)
    g_meta = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor(g),0),1),1)
    b_meta = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor(b_),0),1),1)
    color_scaler = torch.unsqueeze(torch.cat([b_meta, g_meta, r_meta, g_meta], dim=1),2).to(device).float()

    return color_scaler, exp, iso

def get_attention_coords(args):
    
    xs = np.linspace(1, args.img_size//2, num=args.img_size//2)
    ys = np.linspace(1, args.img_size//2, num=args.img_size//2)
    x, y = np.meshgrid(xs, ys, indexing='xy')
    y = np.expand_dims(y, axis=0)
    x = np.expand_dims(x, axis=0)
    coords = np.ascontiguousarray(np.concatenate([x, y]))

    scale = 2 * math.pi
    coords[1,:,:] = coords[1,:,:] / (args.img_size/2) * scale 
    coords[0,:,:] = coords[0,:,:] / (args.img_size/2) * scale
    coords_atten = torch.from_numpy(coords).unsqueeze(0)
    return coords_atten