# MetaISP – Exploiting Global Scene Structure for Accurate Multi-Device Color Rendition [VMV 2023]

## Table of Contents

- [Training](#training)
- [Pre-trained Models](#pretrained)
- [Validation and Metrics Computation](#validation)
- [Single Image Evaluation](#singleeval)
- [Dataset](#dataset)
- [Citation](#cite)

## Training <a name = "training"></a>

In this work, the network is initially pre-trained with monitor-captured data and then fine-tuned with real-world data. The datasets structure can be found [here](#dataset). To pre-trained the network run:

```sh
python train.py --datatype monitor --finetune False --fine_tune_warp False --lr 1e-4 --batch_size 32 --name monitor_pretraining
```

Finally, use the following to fine-tune the network with real-world data:

```sh
python train.py --datatype real --finetune True --pre_path pre/trained/path --fine_tune_warp True --lr 5e-5 --batch_size 8 --name real_finetuning 
```

## Pre-trained Models <a name = "pretrained"></a>
The pre-trained models can be downloaded [here](https://drive.google.com/drive/folders/1tLWlx0LDUjQ9niZje0cfLKt98dx1VEIR?usp=sharing) and should be placed in the folder pre_trained. MetaISPNet_monitor_E and MetaISPNet_real_illuminants are the final results reported in the paper with the all modules. MetaISPNet_real_D does not account for illuminants and iso/exposure module and is used for ablation studies and Xiaomi results in the paper supplementary and zero-shot analysis.


## Validation and Metrics Computation <a name = "validation"></a>

To reproduce the results reported in the paper, run the following command choosing the desired device Pixel 6 Pro, Samsung S22 and IPhone XR (respectively 0,1,2):

```sh
python inference/test.py --full False --pre_path pre_trained/MetaISPNet_real_E.pth --infedev [0,1,2] --iso_exp True
```
For full resolution images change the full flag to True.

To reproduce the illuminants results:
```sh
python inference/test.py --full False --pre_path pre_trained/MetaISPNet_real_illuminants.pth --infedev [0,1,2] --iso_exp True --illuminant True
```
To generate intermediate results interpolating the device's style, turn the flag latent to True, latent_n is the number of intermediate images to be generated, sname is the image name on the dataset that you want to perform the task:
```sh
python inference/test.py --full True --pre_path pre_trained/MetaISPNet_real_E.pth --latent True --latent_n 5 --sname 18
```

The results will be saved in the folder inside /ckpt/name defined by the flag --name.

To compute the metrics run the following command:

```sh
python inference/metrics_compute.py --path_gt datasets/real/device/ --path_pred results/real/device/ \
--meta datasets/real/meta/ --save_name metrics.csv
```

The metrics_compute.py will compute PSNR, SSIM and DeltaE warping the Ground Truth to be aligned with Prediction. --meta is the path to the metadata, which contains the file names.


## Single Image <a name = "singleeval"></a>

To evaluate a single image.

```sh
python inference/single_image_test.py --path_model --path_image --device [xiaomi, iphone] --iso_exp [True,False] --illuminant [True,False]
```

Different devices like xiaomi may underperform due to iso and exp too different from the ones trained with iPhone. In the paper, we used the model pre_trained/MetaISPNet_real_D without iso/exp module to have consistent images. In case will want to use the iso/exp module be aware that we clipped it in the code when Xiaomi is select in order to make it in similar range as the iPhone XR images.

If you want to perform inference with RAW images from different devices, you just need to adjust things like the bayer pattern structure, iso/exp range, black level and so on.

## Dataset <a name = "dataset"></a>

The datasets can be found [here](https://drive.google.com/drive/folders/1tLWlx0LDUjQ9niZje0cfLKt98dx1VEIR?usp=sharing). To reproduce our results you just need to download and place it in the root folder.
```datasets/
├── real/
│   ├── iphone/
│   │   ├── bilinear/
│   │   ├── full/
│   │   ├── raw/
│   │   └── rgb/
│   ├── samsung/
│   │   ├── full/
│   │   └── rgb/
│   ├── pixel/
│   │   ├── full/
│   │   └── rgb/
│   └── meta/
│       ├── iphone/
│       ├── samsung/        
│       └── pixel/  
└── monitor/
    ├── iphone/
    │   ├── bilinear/
    │   ├── full/
    │   ├── raw/
    │   └── rgb/
    ├── samsung/
    │   ├── full/
    │   └── rgb/
    ├── pixel/
    │   ├── full/
    │   └── rgb/
    └── meta/
        ├── iphone/
        ├── samsung/        
        └── pixel/  
```

## Citation
```
@inproceedings {10.2312:vmv.20231236,
      booktitle = {Vision, Modeling, and Visualization},
      editor = {Guthe, Michael and Grosch, Thorsten},
      title = {{MetaISP -- Exploiting Global Scene Structure for Accurate Multi-Device Color Rendition}},
      author = {Souza, Matheus and Heidrich, Wolfgang},
      year = {2023},
      publisher = {The Eurographics Association},
      ISBN = {978-3-03868-232-5},
      DOI = {10.2312/vmv.20231236}
}
```
## Aknowledgements
This code is heavily based on LiteISP and XCIT. We thank the authors for making their code available.
