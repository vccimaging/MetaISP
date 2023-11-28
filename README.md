# MetaISP – Exploiting Global Scene Structure for Accurate Multi-Device Color Rendition [VMV 2023]

## Table of Contents

- [Training](#training)
- [Validation and Metrics Computation](#validation)
- [Single Image Evaluation](#singleeval)
- [Dataset](#dataset)
- [Citation](#cite)

## Training <a name = "training"></a>

In this work, the network is initially pre-trained with monitor-captured data and then fine-tuned with real-world data. The datasets structure can be found [here](#dataset). To pre-trained the network run:

```sh
   python train.py --datatype monitor --finetune False --fine_tune_warp False \
   --lr 1e-4 --batch_size 32 --name monitor_pretraining
```

Finally, use the following to fine-tune the network with real-world data:

```sh
      python train.py --datatype real --finetune True --pre_path pre/trained/path \
      --fine_tune_warp True --lr 5e-5 --batch_size 8 --name real_finetuning 
```

## Validation and Metrics Computation <a name = "validation"></a>

## Single Image Evaluation <a name = "singleeval"></a>

## Dataset <a name = "dataset"></a>

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

## Contact
