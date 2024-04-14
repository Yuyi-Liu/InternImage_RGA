# InternImage_RGA for Semantic Segmentation

This folder contains the implementation of the InternImage with robust geometry-adaptive convolution for semantic segmentation. 

Our segmentation code is developed on top of [InternImage](https://github.com/OpenGVLab/InternImage) and [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/Yuyi-Liu/InternImage_RGA.git
cd InternImage_RGA
```

- Create a conda virtual environment and activate it:

```bash
conda create -n internimage python=3.7 -y
conda activate internimage
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install torch==1.11 with CUDA==11.3 and nvcc:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge cudatoolkit-dev=11.3 -y # to install nvcc
```

- Install other requirements:

  note: conda opencv will break torchvision as not to support GPU, so we need to install opencv using pip. 	  

```bash
conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
pip install opencv-python
```

- Install `timm` and `mmcv-full`:

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Install `mmsegmentation`:

```bash
cd ..
git clone https://github.com/Yuyi-Liu/mmsegmentation_RGA.git
cd mmsegmentation_RGA
git checkout RGA-v0.27.0
pip install -v -e .
```

- Compile CUDA operators
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
- You can also install the operator using .whl files
[DCNv3-1.0-whl](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files)

### Datasets

Download [NYUDepthv2](https://drive.google.com/file/d/18aV2E7--ZmS53aA5zE2zDbRZi0_UgeN8/view?usp=drive_link) and [SUNRGBD](https://drive.google.com/file/d/1MBtcr3Pxi3wDjFVOOMzIJlZ-ucNnXZmA/view?usp=drive_link) datasets.

Prepare datasets according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.


### Evaluation

To evaluate our `InternImage` on ADE20K val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval mIoU
```
You can download checkpoint files from [here](https://huggingface.co/OpenGVLab/InternImage/tree/fc1e4e7e01c3e7a39a3875bdebb6577a7256ff91). Then place it to segmentation/checkpoint_dir/seg.

For example, to evaluate the `InternImage-T` with a single GPU:

```bash
python test.py configs/ade20k/upernet_internimage_t_512_160k_ade20k.py checkpoint_dir/seg/upernet_internimage_t_512_160k_ade20k.pth --eval mIoU
```

For example, to evaluate the `InternImage-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/ade20k/upernet_internimage_b_512_160k_ade20k.py checkpoint_dir/seg/upernet_internimage_b_512_160k_ade20k.pth 8 --eval mIoU
```

### Training

To train an `InternImage` on ADE20K, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `InternImage-T` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/ade20k/upernet_internimage_t_512_160k_ade20k.py 8
```

### Modify the insertion location of RGA layer

Please refer to the following code in the segmentation/mmseg_custom/models/backbones/intern_image.py file to modify the insertion location.

```python3
class InternImageLayer(nn.Module):
    def __init__(self,idx,
                 core_op,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False): # for InternImage-H/G
        # ........
        if self.idx not in [100,101]:
            self.dcn = core_op(
                channels=channels,
                kernel_size=3,
                stride=1,
                pad=1,
                dilation=1,
                group=groups,
                offset_scale=offset_scale,
                act_layer=act_layer,
                norm_layer=norm_layer,
                dw_kernel_size=dw_kernel_size, # for InternImage-H/G
                center_feature_scale=center_feature_scale) # for InternImage-H/G
      # .........
```

