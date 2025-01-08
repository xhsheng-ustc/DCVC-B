# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import concurrent.futures
import json
import multiprocessing
import time

import torch
import torch.nn.functional as F
import numpy as np
from src.models.video_model_hierarchical import DMC
from src.models.image_model import IntraNoAR
from src.utils.common import str2bool, create_folder, generate_log_json, dump_json
from src.utils.stream_helper import get_padding_size, get_state_dict
from src.utils.video_reader import PNGReader
from src.utils.video_writer import PNGWriter
from src.utils.coding_structure_RA_test import get_coding_structure_RA_test,empty_memory
from tqdm import tqdm
from pytorch_msssim import ms_ssim

def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument("--ec_thread", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--stream_part_i", type=int, default=1)
    parser.add_argument("--stream_part_b", type=int, default=1)
    parser.add_argument('--i_frame_model_path', type=str)
    parser.add_argument('--b_frame_model_path',  type=str)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--i_frame_q_indexes', type=int, nargs="+")
    parser.add_argument('--b_frame_q_indexes', type=int, nargs="+")
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--cuda_device", default=None,
                        help="the cuda device used, e.g., 0; 0,1; 1,2,3; etc.")
    parser.add_argument('--calc_ssim', type=str2bool, default=False, required=False)
    parser.add_argument('--write_stream', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--decoded_frame_path', type=str, default='decoded_frames')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    return args


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def run_test(b_frame_net, i_frame_net, args):
    frame_num = args['frame_num']
    gop_size = args['gop_size']
    write_stream = 'write_stream' in args and args['write_stream']
    save_decoded_frame = 'save_decoded_frame' in args and args['save_decoded_frame']
    verbose = args['verbose'] if 'verbose' in args else 0
    device = next(i_frame_net.parameters()).device

    if args['src_type'] == 'png':
        src_reader = PNGReader(args['src_path'], args['src_width'], args['src_height'])

    if save_decoded_frame:
        if args['src_type'] == 'png':
            recon_writer = PNGWriter(args['recon_path'], args['src_width'], args['src_height'])

    frame_types = []
    psnrs = []
    msssims= []

    bits = []
    frame_pixel_num = 0

    start_time = time.time()
    b_frame_number = 0
    overall_b_encoding_time = 0
    overall_b_decoding_time = 0
    coding_order_list, forward_ref_lists, backword_ref_lists, layer_id_lists = get_coding_structure_RA_test()
    dpb_list = []
    y_rec_list = []
    uv_rec_list = []
    x_hat_list = []
    for _ in range(frame_num):
        frame_types.append(0)
        psnrs.append(0)
        msssims.append(0)
        bits.append(0)
        dpb_list.append(
            {
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None
            }
        )
        y_rec_list.append(0)
        uv_rec_list.append(0)
        x_hat_list.append(0)

    with torch.no_grad():
        for coding_idx in range(frame_num):
            frame_start_time = time.time()
            frame_idx = coding_order_list[coding_idx]
            dpb_list = empty_memory(dpb_list,frame_idx)
            forward_ref = forward_ref_lists[coding_idx]
            backword_ref = backword_ref_lists[coding_idx]
            layer_id = layer_id_lists[coding_idx]-1
            # if layer_id == 4 or layer_id == 3:
                # layer_id=2
            # if layer_id == 4 or layer_id==5:
            if layer_id == 4:
                layer_id=3
            rgb = src_reader.read_one_frame(frame_idx+1, dst_format="rgb")
            x = np_image_to_tensor(rgb)
            x = x.to(device)
            pic_height = x.shape[2]
            pic_width = x.shape[3]

            if frame_pixel_num == 0:
                frame_pixel_num = x.shape[2] * x.shape[3]
            else:
                assert frame_pixel_num == x.shape[2] * x.shape[3]

            # pad if necessary
            padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, 16)
            x_padded = torch.nn.functional.pad(
                x,
                (padding_l, padding_r, padding_t, padding_b),
                mode="replicate",
            )

            bin_path = os.path.join(args['bin_folder'], f"{frame_idx}.bin") \
                if write_stream else None

            if frame_idx % gop_size == 0:
                result = i_frame_net.encode_decode(x_padded, args['q_in_ckpt'],
                                                   args['i_frame_q_index'], bin_path,
                                                   pic_height=pic_height, pic_width=pic_width)
                dpb_list[frame_idx]['ref_frame'] = result["x_hat"]
                recon_frame = result["x_hat"]
                frame_types[frame_idx]=0
                bits[frame_idx]=result["bit"]
            else:
                dpb_l = dpb_list[forward_ref[0]]
                dpb_r = dpb_list[backword_ref[0]]
                result = b_frame_net.encode_decode(x_padded, dpb_l, dpb_r, layer_id, args['q_in_ckpt'],
                                                   args['i_frame_q_index'], bin_path,
                                                   pic_height=pic_height, pic_width=pic_width)
                dpb = result["dpb"]
                dpb_list[frame_idx] = dpb
                recon_frame = dpb["ref_frame"]
                frame_types[frame_idx]=1
                bits[frame_idx]=result["bit"]
                b_frame_number += 1
                overall_b_encoding_time += result['encoding_time']
                overall_b_decoding_time += result['decoding_time']

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            x_hat_list[frame_idx]=x_hat
            psnr = PSNR(x_hat, x)
            if args['calc_ssim']:
                msssim = ms_ssim(x_hat, x, data_range=1).item()
            else:
                msssim = 0.
            psnrs[frame_idx]=psnr
            msssims[frame_idx]=msssim
            frame_end_time = time.time()

            if verbose >= 2:
                print(f"frame {frame_idx}, {frame_end_time - frame_start_time:.3f} seconds,",
                      f"bits: {bits[frame_idx]:.3f}, PSNR: {psnrs[frame_idx]:.4f}, MS-SSIM: {msssims[frame_idx]:.4f} ")

        if save_decoded_frame:
            for frame_idx in range(frame_num):
                rgb_np = x_hat_list[frame_idx].squeeze(0).cpu().numpy()
                recon_writer.write_one_frame(rgb=rgb_np, src_format='rgb')

    if save_decoded_frame:
        avg_bpp = sum(bits) / len(bits) / pic_width / pic_height
        avg_psnr = sum(psnrs) / len(psnrs)
        folder_name = f"{args['rate_idx']}_{avg_bpp:.4f}_{avg_psnr:.4f}"
        os.rename(args['recon_path'], args['recon_path'] + f'/../{folder_name}')

    test_time = time.time() - start_time
    if verbose >= 1 and b_frame_number > 0:
        print(f"encoding/decoding {b_frame_number} B frames, "
              f"average encoding time {overall_b_encoding_time/b_frame_number * 1000:.0f} ms, "
              f"average decoding time {overall_b_decoding_time/b_frame_number * 1000:.0f} ms.")
    log_result = generate_log_json(frame_num, frame_pixel_num, test_time,
                                    frame_types, bits, psnrs, msssims, verbose=True)
    return log_result


i_frame_net = None  # the model is initialized after each process is spawn, thus OK for multiprocess
b_frame_net = None


def encode_one(args):
    global i_frame_net
    global b_frame_net

    sub_dir_name = args['video_path']
    bin_folder = os.path.join(args['stream_path'], sub_dir_name, str(args['rate_idx']))
    if args['write_stream']:
        create_folder(bin_folder, True)

    if args['save_decoded_frame']:
        recon_path = os.path.join(args['decoded_frame_path'], sub_dir_name, str(args['rate_idx']))
        create_folder(recon_path)
    else:
        recon_path = None

    args['src_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['bin_folder'] = bin_folder
    args['recon_path'] = recon_path

    result = run_test(b_frame_net, i_frame_net, args)

    result['ds_name'] = args['ds_name']
    result['video_path'] = args['video_path']
    result['rate_idx'] = args['rate_idx']

    return result


def worker(args):
    return encode_one(args)


def init_func(args):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    gpu_num = 0
    if args.cuda:
        gpu_num = torch.cuda.device_count()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    global i_frame_net
    i_state_dict = get_state_dict(args.i_frame_model_path)
    i_frame_net = IntraNoAR(ec_thread=args.ec_thread, stream_part=args.stream_part_i,
                            inplace=True)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    global b_frame_net
    b_state_dict = get_state_dict(args.b_frame_model_path)
    b_frame_net = DMC(ec_thread=args.ec_thread, stream_part=args.stream_part_b,
                        inplace=True)
    b_frame_net.load_state_dict(b_state_dict,False)
    b_frame_net = b_frame_net.to(device)
    b_frame_net.eval()

    if args.write_stream:
        if b_frame_net is not None:
            b_frame_net.update(force=True)
        i_frame_net.update(force=True)


def main():
    begin_time = time.time()

    torch.backends.cudnn.enabled = True
    args = parse_args()

    if args.cuda_device is not None and args.cuda_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num,
                                                                 initializer=init_func,
                                                                 initargs=(args,))
    objs = []

    count_frames = 0
    count_sequences = 0

    rate_num = args.rate_num
    i_frame_q_scale_enc, i_frame_q_scale_dec = \
        IntraNoAR.get_q_scales_from_ckpt(args.i_frame_model_path)
    print("q_scale_enc in intra ckpt: ", end='')
    for q in i_frame_q_scale_enc:
        print(f"{q:.3f}, ", end='')
    print()
    print("q_scale_dec in intra ckpt: ", end='')
    for q in i_frame_q_scale_dec:
        print(f"{q:.3f}, ", end='')
    print()
    i_frame_q_indexes = []
    q_in_ckpt = False
    if args.i_frame_q_indexes is not None:
        assert len(args.i_frame_q_indexes) == rate_num
        i_frame_q_indexes = args.i_frame_q_indexes
    elif len(i_frame_q_scale_enc) == rate_num:
        assert rate_num == 4
        q_in_ckpt = True
        i_frame_q_indexes = [0, 1, 2, 3]
    else:
        assert rate_num >= 2 and rate_num <= 64
        for i in np.linspace(0, 63, num=rate_num):
            i_frame_q_indexes.append(int(i+0.5))


    y_q_basic_enc, y_q_basic_dec, y_q_scale_enc, y_q_scale_dec, mv_y_q_basic_enc, mv_y_q_basic_dec, mv_y_q_scale_enc, mv_y_q_scale_dec = \
        DMC.get_q_scales_from_ckpt(args.b_frame_model_path)
    print("y_q_basic_enc in inter ckpt: ", end='')
    for q in y_q_basic_enc:
        print(f"{q:.3f}, ", end='')
    print()
    print("y_q_basic_dec in inter ckpt: ", end='')
    for q in y_q_basic_dec:
        print(f"{q:.3f}, ", end='')
    print()
    print("y_q_scale_enc in inter ckpt: ", end='')
    for q in y_q_scale_enc:
        print(f"{q:.3f}, ", end='')
    print()
    print("y_q_scale_dec in inter ckpt: ", end='')
    for q in y_q_scale_dec:
        print(f"{q:.3f}, ", end='')
    print()
    print("mv_y_q_basic_enc in inter ckpt: ", end='')
    for q in mv_y_q_basic_enc:
        print(f"{q:.3f}, ", end='')
    print()
    print("mv_y_q_basic_dec in inter ckpt: ", end='')
    for q in mv_y_q_basic_dec:
        print(f"{q:.3f}, ", end='')
    print()
    print("mv_y_q_scale_enc in inter ckpt: ", end='')
    for q in mv_y_q_scale_enc:
        print(f"{q:.3f}, ", end='')
    print()
    print("mv_y_q_scale_dec in inter ckpt: ", end='')
    for q in mv_y_q_scale_dec:
        print(f"{q:.3f}, ", end='')
    print()

    b_frame_q_indexes = i_frame_q_indexes

    print(f"testing {rate_num} rates, using q_indexes: ", end='')
    for q in i_frame_q_indexes:
        print(f"{q}, ", end='')
    print()

    root_path = config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq_name in config[ds_name]['sequences']:
            count_sequences += 1
            for rate_idx in range(rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['q_in_ckpt'] = q_in_ckpt
                cur_args['i_frame_q_index'] = i_frame_q_indexes[rate_idx]
                cur_args['b_frame_q_index'] = b_frame_q_indexes[rate_idx]
                cur_args['video_path'] = seq_name
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq_name]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq_name]['width']
                cur_args['gop_size'] = config[ds_name]['sequences'][seq_name]['gop']
                cur_args['frame_num'] = config[ds_name]['sequences'][seq_name]['frames']
                cur_args['calc_ssim'] = args.calc_ssim
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['write_stream'] = args.write_stream
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['decoded_frame_path'] = f'{args.decoded_frame_path}'
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose

                count_frames += cur_args['frame_num']

                obj = threadpool_executor.submit(worker, cur_args)
                objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}
            for rate in range(rate_num):
                for res in results:
                    if res['rate_idx'] == rate and ds_name == res['ds_name'] \
                            and seq == res['video_path']:
                        log_result[ds_name][seq][f"{rate:03d}"] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
