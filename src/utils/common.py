import json
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np


def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def scale_list_to_str(scales):
    s = ''
    for scale in scales:
        s += f'{scale:.2f} '

    return s


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")


@patch('json.encoder.c_make_encoder', None)
def dump_json(obj, fid, float_digits=-1, **kwargs):
    of = json.encoder._make_iterencode  # pylint: disable=W0212

    def inner(*args, **kwargs):
        args = list(args)
        # fifth argument is float formater which we will replace
        args[4] = lambda o: format(o, '.%df' % float_digits)
        return of(*args, **kwargs)

    with patch('json.encoder._make_iterencode', wraps=inner):
        json.dump(obj, fid, **kwargs)


def generate_log_json(frame_num, frame_pixel_num, test_time, frame_types, bits, psnrs, ssims, verbose=False):
    i_bits = 0
    i_psnr = 0
    i_ssim = 0
    b_bits = 0
    b_psnr = 0
    b_ssim = 0
    i_num = 0
    b_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            i_bits += bits[idx]
            i_psnr += psnrs[idx]
            i_ssim += ssims[idx]
            i_num += 1
        else:
            b_bits += bits[idx]
            b_psnr += psnrs[idx]
            b_ssim += ssims[idx]
            b_num += 1

    log_result = {}
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['i_frame_num'] = i_num
    log_result['b_frame_num'] = b_num
    log_result['ave_i_frame_bpp'] = i_bits / i_num / frame_pixel_num
    log_result['ave_i_frame_psnr'] = i_psnr / i_num
    log_result['ave_i_frame_msssim'] = i_ssim / i_num
    if verbose:
        log_result['frame_bpp'] = list(np.array(bits) / frame_pixel_num)
        log_result['frame_psnr'] = psnrs
        log_result['frame_msssim'] = ssims
        log_result['frame_type'] = frame_types
    log_result['test_time'] = test_time
    if b_num > 0:
        total_b_pixel_num = b_num * frame_pixel_num
        log_result['ave_b_frame_bpp'] = b_bits / total_b_pixel_num
        log_result['ave_b_frame_psnr'] = b_psnr / b_num
        log_result['ave_b_frame_msssim'] = b_ssim / b_num
    else:
        log_result['ave_b_frame_bpp'] = 0
        log_result['ave_b_frame_psnr'] = 0
        log_result['ave_b_frame_msssim'] = 0
    log_result['ave_all_frame_bpp'] = (i_bits + b_bits) / (frame_num * frame_pixel_num)
    log_result['ave_all_frame_psnr'] = (i_psnr + b_psnr) / frame_num
    log_result['ave_all_frame_msssim'] = (i_ssim + b_ssim) / frame_num

    return log_result
