# Explore In-Context Segmentation via Latent Diffusion Models


<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2403.09616-b31b1b.svg)](https://arxiv.org/abs/2403.09616)
[![Project Website](https://img.shields.io/badge/🔗-Project_Website-blue.svg)](https://wang-chaoyang.github.io/project/refldmseg)

</div>

<div>
  <p align="center" style="font-size: larger;">
    <strong>AAAI 2025</strong>
  </p>
</div>

<p align="center">
<img src="assets/teaser.png" width=95%>
<p>


## Requirements

1. Install `torch==2.1.0`.
2. Install pip packages via `pip install -r requirements.txt` and [alpha_clip](https://github.com/SunzeY/AlphaCLIP).
3. Our model is based on [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5), download and put it into `datasets/pretrain`. Put the checkpoints of [alpha_clip](https://github.com/SunzeY/AlphaCLIP) into `datasets/pretrain/alpha-clip`.

## Data Preparation

Please download the following datasets: [COCO 2014](https://cocodataset.org/#download), [DAVIS16](https://davischallenge.org/davis2016/code.html), [VSPW](https://github.com/sssdddwww2/vspw_dataset_download), and PASCAL, which includes [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html). And then download the [meta files](https://huggingface.co/chaoyangw/ldis/tree/main/datasets). Put them under `datasets` and rearrange as follows.

```
datasets
├── pascal
│   ├── JPEGImages
│   ├── SegmentationClassAug
│   └── metas
├── davis16
│   ├── JPEGImages
│   ├── Annotations
│   └── metas
├── vspw
│   ├── images
│   ├── masks
│   └── metas
└── coco20i
    ├── annotations
    │   ├── train2014
    │   └── val2014
    ├── metas
    ├── train2014
    └── val2014
```

## Train

The codes in [scripts](scripts) is launched by accelerate. The saved path is specified by `--output_dir` defined in [args](module/utils/prepare_args.py).

```
# ldis1
accelerate launch --multi_gpu --num_processes [GPUS] scripts/modelf.py --config configs/cfg.py
# ldisn
accelerate launch --multi_gpu --num_processes [GPUS] scripts/modeln.py --config configs/cfg.py --mask_alpha 0.4
```

## Inference

```
# ldis1
accelerate launch --multi_gpu --num_processes [GPUS] scripts/modelf.py --config configs/cfg.py --only_val 1 --val_dataset pascal --output_dir [the path of ckpt]
# ldisn
accelerate launch --multi_gpu --num_processes [GPUS] scripts/modeln.py --config configs/cfg.py --only_val 1 --val_dataset pascal --output_dir [the path of ckpt] --mask_alpha 0.4
```
The pretrained models can be found [here](https://huggingface.co/chaoyangw/ldis/tree/main/weights).

## Citation

If you find our work useful, please kindly consider citing our paper:

```bibtex
@article{wang2024explore,
  title={Explore In-Context Segmentation via Latent Diffusion Models},
  author={Wang, Chaoyang and Li, Xiangtai and Ding, Henghui and Qi, Lu and Zhang, Jiangning and Tong, Yunhai and Loy, Chen Change and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2403.09616},
  year={2024}
}
```

## License

MIT license