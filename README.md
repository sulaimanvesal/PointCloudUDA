# Adapt Everywhere: Unsupervised Adaptation of Point-Clouds and Entropy Minimisation for Multi-modal Cardiac-MR Segmentation

Pytorch implementation of our method for unsupervised domain adaptation for cardiac-MR using Point-cloud and Entropy manimisation. Based on this implementation, our result achieved state-of-the-art UDA performance for MULTI-SEQUENCE CARDIAC MR SEGMENTATION CHALLENGE, MICCAI-STACOM 2019.

### Contact: Sulaiman Vesal (sulaiman dot vesal at fau dot com)

---
# Abstract

>Deep learning models are sensitive to domain shift phenomena. A model trained on one set of images can't generalise well when tested on images from a different domain or modality, despite imaging similar anatomical structures. It is because the data distribution between the two domains is different. Moreover, creating annotation for every new imaging modality is a tedious and time-consuming task, which also suffers from high inter and intra- observer variability. The recent unsupervised domain adaptation (UDA) methods intend to reduce the gap between source and target domains by using entirely unlabelled target domain data and labelled source domain data. However, current state-of-the-art (SOTA) UDA methods demonstrate degraded performance when there is insufficient data in source and target domains.  In this paper, we propose a UDA method for multi-modal Cardiac Magnetic Resonance (CMR) image segmentation. The proposed method is based on adversarial learning and adapts network features between source and target domain in different spaces. The paper introduces an end-to-end framework that integrates: a) entropy minimisation, b) output feature space alignment and c) novel Point-cloud shape adaptation based on latent features learned by the segmentation model. We validated our method on the publicly available multi-sequence cardiac dataset by adapting from the annotated source domain (3D balanced- Steady-State Free Procession-MRI) to the unannotated target domain (3D Late-gadolinium enhance-MRI). The experimental results highlight that by enforcing adversarial learning in different parts of the network, the proposed method outperforms the SOTA methods in the same setting, and significantly diminishes the data distribution gap.
---
<img src="https://github.com/sulaimanvesal/PointCloudUDA/blob/master/images/git_framework.png" width="800">

## Dataset
* Download the Multi-sequence Cardiac MR Segmentation Challenge (MS-CMRSeg 2019) dataset: 
      https://zmiclab.github.io/projects/mscmrseg19/
      
* Download the MM-WHS: Multi-Modality Whole Heart Segmentation Challenge (MM-WHS 2018) dataset: 
      https://zmiclab.github.io/projects/mmwhs/
      
    *  The pre-processed data has been released from our work [PnP-AdaNet](https://github.com/cchen-cc/SIFA).     
    *  The training data can be downloaded here. The testing CT data can be downloaded here. The testing MR data can be downloaded here.
    *  Put tfrecord data of two domains into corresponding folders under ./data accordingly.
    *  Run ./create_datalist.py to generate the datalists containing the path of each data.


## Installation
    Install PyTorch 1.4 + CUDA 10.0 
    Clone this repo.
    
```
git clone https://github.com/sulaimanvesal/PointCloudUDA/
cd PointCloudUDA
```
## Train

    Modify paramter values in ./config_param.json
    Run ./main.py to start the training process

## Evaluate
    Specify the model path and test file path in ./evaluate.py
    Run ./evaluate.py to start the evaluation.


---
## Citations
Please consider citing the following papers in your publications if they help your research.
```
@ARTICLE{9302580,
  author={S. {Vesal} and M. {Gu}  and R. {Kosti} and A. {Maier} and N. {Ravikumar}},
  journal={IEEE Transaction in Medical Imaging}, 
  title={Adapt Everywhere: Unsupervised Adaptation of Point-Clouds and Entropy Minimisation for Multi-modal Cardiac-MR Segmentation}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JBHI.2020.3046449}}
```
