import os

def test_one_model():
    output_json_path = f"DCVC-B-msssim-results.json"
    image_model_path = './checkpoints/cvpr2023_image_ssim.pth.tar'
    command_line = (" python test_video_hierarchical.py"
                    f" --i_frame_model_path  {image_model_path}"
                    f" --b_frame_model_path ./checkpoints/DCVC-B-msssim-model.pth.tar"
                    " --rate_num 4"
                    " --verbose 2"
                    " --calc_ssim 1 "
                    " --test_config recommended_test_full_results_IP32.json --cuda 1 -w 1 "
                    f" --output_path {output_json_path} --save_decoded_frame 0 --decoded_frame_path ./rec/"
                    f" --write_stream 0 --stream_path ./bin/")

    print(command_line)
    os.system(command_line)

def main():
    test_one_model()


if __name__ == "__main__":
    main()