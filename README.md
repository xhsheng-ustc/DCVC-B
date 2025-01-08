# Introduction

Official Pytorch implementation for DCVC-B: [Bi-Directional Deep Contextual Video Compression](https://arxiv.org/abs/2408.08604), in TMM 2025.

# Prerequisites
* Python 3.8 and conda, get [Conda](https://www.anaconda.com/)
* CUDA if want to use GPU
* Environment
    ```
    conda create -n $YOUR_PY38_ENV_NAME python=3.8
    conda activate $YOUR_PY38_ENV_NAME

    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    pip install -r requirements.txt
    ```

# Build the project
Please build the C++ code if want to test with actual bitstream writing. There is minor difference about the bits for calculating the bits using entropy (the method used in the paper to report numbers) and actual bitstreaming writing. There is overhead when writing the bitstream into the file and the difference percentage depends on the bitstream size. Usually, the overhead for 1080p content is less than 0.5%.
## On Windows
```bash
cd src
mkdir build
cd build
conda activate $YOUR_PY38_ENV_NAME
cmake ../cpp -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

## On Linux
```bash
sudo apt-get install cmake g++
cd src
mkdir build
cd build
conda activate $YOUR_PY38_ENV_NAME
cmake ../cpp -DCMAKE_BUILD_TYPE=Release
make -j
```

# Pretrained models

* Download [Our B-frame pretrained models](https://drive.google.com/drive/folders/1o6xOuKMvousSwx39S-hbs7EYNaTr8vAT) and put them into ./checkpoints folder.
* Download [We use the same image model as DCVC-DC](https://1drv.ms/u/s!AozfVVwtWWYoiWdwDhEkZMIfpon5?e=JcGri5) and put them into ./checkpoints folder.

# Test the models

Example to test pretrained model with four rate points:
```bash
python test_all_video_psnr.py
```
```bash
python test_all_video_msssim.py
```
# Acknowledgement
The implementation is based on [DCVC-DC](https://github.com/microsoft/DCVC/tree/main/DCVC-DC) and [DCVC-SDD](https://github.com/xhsheng-ustc/DCVC-SDD/tree/main).
# Citation
If you find this work useful for your research, please cite:

```
@article{sheng2024bi,
  title={Bi-Directional Deep Contextual Video Compression},
  author={Sheng, Xihua and Li, Li and Liu, Dong and Wang, Shiqi},
  journal={arXiv preprint arXiv:2408.08604},
  year={2024}
}
```

```
@article{sheng2024spatial,
  title={Spatial Decomposition and Temporal Fusion based Inter Prediction for Learned Video Compression},
  author={Sheng, Xihua and Li, Li and Liu, Dong and Li, Houqiang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```
```
@inproceedings{li2023neural,
  title={Neural Video Compression with Diverse Contexts},
  author={Li, Jiahao and Li, Bin and Lu, Yan},
  booktitle={{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
             {CVPR} 2023, Vancouver, Canada, June 18-22, 2023},
  year={2023}
}
```
