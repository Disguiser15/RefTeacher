# RefTeacher

[![Python](https://img.shields.io/badge/python-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-%237732a8)
[![MAC-Group](https://img.shields.io/badge/mac-group-orange.svg)](https://mac.xmu.edu.cn/)

RefTeacher is a strong baseline method for Semi-Supervised Referring Expression Comprehension. This is the official implementations of our paper  "RefTeacher: A Strong Baseline for Semi-Supervised Referring Expression Comprehension", including models such as [RealGIN](https://ieeexplore.ieee.org/abstract/document/9470913) and [SimREC](https://arxiv.org/abs/2204.07913).


<p align="center">
	<img src="./misc/RefTeacher.jpg" width="600">
</p>


## Updates
- (2022/11/28) Release our RefTeacher project.

## Installation
- Clone this repo
```bash
git clone https://github.com/Disguiser15/RefTeacher.git
cd RefTeacher
```
- Create a conda virtual environment and activate it
```bash
conda create -n refteacher python=3.7 -y
conda activate refteacher
```
- Install Pytorch following the [official installation instructions](https://pytorch.org/get-started/locally/)
- Install mmcv following the [installation guide](https://github.com/open-mmlab/mmcv#installation)
- Install [Spacy](https://spacy.io/) and initialize the [GloVe](https://github-releases.githubusercontent.com/84940268/9f4d5680-4fed-11e9-9dd2-988cce16be55?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210815%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210815T072922Z&X-Amz-Expires=300&X-Amz-Signature=1bd1bd4fc52057d8ac9eec7720e3dd333e63c234abead471c2df720fb8f04597&X-Amz-SignedHeaders=host&actor_id=48727989&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Den_vectors_web_lg-2.1.0.tar.gz&response-content-type=application%2Foctet-stream) and install other requirements as follows:
```
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```

## Data preparation

-  Follow the instructions of  [DATA_PRE_README.md](./DATA_PRE_README.md) to generate training data and testing data.
-  Download the pretrained weights of backbone (vgg, darknet, cspdarknet, DResNet, etc.).  Expect for DResNet, all pretrained backbones are trained on COCO 2014 *train+val*  set while removing the images appeared in the *val+test* sets of RefCOCO, RefCOCO+ and RefCOCOg (nearly 6500 images).  Please follow the instructions of  [DATA_PRE_README.md](./DATA_PRE_README.md) to download them.

## Training and Evaluation 

<!-- 1. Prepare your settings. To train a model, you should  modify ``./config/config.yaml``  to adjust the settings  you want. 
2. Train the model. run ` train.py`  under the main folder to start training:
```
python train.py --config ./config/config.yaml
```
3. Test the model.   Then, you can run ` test.py`  by
```
python test.py --eval-weights ./logs/simrec/1/weights/det_best.pth
```
4. Training log.  Logs are stored in ``./logs`` directory, which records the detailed training curve and accuracy per epoch. If you want to log the visualizations, please  set  ``LOG_IMAGE`` to ``True`` in ``config.yaml``.    -->

1. **Config preparation**. Prepare your own configs in [configs](./config). We provided config files including [Supervised baseline](./config/sup), [STAC baseline](./config/stac) and [RefTeacher](./config/refteacher).

2. **Train the model**. Run `bash/train.sh` under the corresponding directory in [scripts](./scripts).
- Train Supervised baseline.
```shell
bash scripts/sup/train.sh config/sup/realgin_sup_baseline.yaml None
```
- Train STAC baseline.
```shell
bash scripts/stac/train.sh config/stac/realgin_semi_stac.yaml /path/to/pretrained/checkpoint.pth
```
Setting `/path/to/pretrained/checkpoint.pth` to the specific `checkpoint.pth` you want to load as the teacher model.
- Train our RefTeacher.
```shell
# RealGIN
bash scripts/refteacher/train.sh config/refteacher/realgin_semi_refteacher.yaml
# SimREC
bash scripts/refteacher/train.sh config/refteacher/simrec_semi_refteacher.yaml
```
The `training logs`, `tensorboard logs` and `model checkpoints` will be automatically saved under `LOG_PATH` set in [config](./config).

3. **Test the model.** Run `bash/eval.sh` in [scripts](./scripts/eval) to evaluate the saved teacher or student weight.
```shell
# RealGIN
bash scripts/eval/eval.sh config/refteacher/realgin_semi_refteacher.yaml /path/to/checkpoint
# SimREC
bash scripts/eval/eval.sh config/refteacher/simrec_semi_refteacher.yaml /path/to/checkpoint
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If this repository is helpful for your research, or you want to refer the provided results in your paper, consider cite:
```BibTeX
@article{luo2022what,
  title={What Goes beyond Multi-modal Fusion in One-stage Referring Expression Comprehension: An Empirical Study},
  author={Luo, Gen and Zhou, Yiyi and Sun, Jiamu and Huang, Shubin and Sun, Xiaoshuai and Ye, Qixiang and Wu, Yongjian and Ji, Rongrong},
  journal={arXiv preprint arXiv:2204.07913},
  year={2022}
}
```
```BibTeX
@article{zhu2022seqtr,
  title={SeqTR: A Simple yet Universal Network for Visual Grounding},
  author={Zhu, Chaoyang and Zhou, Yiyi and Shen, Yunhang and Luo, Gen and Pan, Xingjia and Lin, Mingbao and Chen, Chao and Cao, Liujuan and Sun, Xiaoshuai and Ji, Rongrong},
  journal={arXiv preprint arXiv:2203.16265},
  year={2022}
}
```
```BibTeX
@InProceedings{Luo_2020_CVPR,
author = {Luo, Gen and Zhou, Yiyi and Sun, Xiaoshuai and Cao, Liujuan and Wu, Chenglin and Deng, Cheng and Ji, Rongrong},
title = {Multi-Task Collaborative Network for Joint Referring Expression Comprehension and Segmentation},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
```BibTeX
@inproceedings{luo2020cascade,
  title={Cascade grouped attention network for referring expression segmentation},
  author={Luo, Gen and Zhou, Yiyi and Ji, Rongrong and Sun, Xiaoshuai and Su, Jinsong and Lin, Chia-Wen and Tian, Qi},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={1274--1282},
  year={2020}
}
```
```BibTeX
@article{zhou2021real,
  title={A real-time global inference network for one-stage referring expression comprehension},
  author={Zhou, Yiyi and Ji, Rongrong and Luo, Gen and Sun, Xiaoshuai and Su, Jinsong and Ding, Xinghao and Lin, Chia-Wen and Tian, Qi},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
}
```
```BibTeX
@misc{luo2022simrec,
  author =       {Gen Luo and Tianhe Ren},
  title =        {SimREC},
  howpublished = {\url{https://github.com/luogen1996/SimREC}},
  year =         {2022}
}
```
## Acknowledgement

Thanks a lot for the nicely organized code from the following repos
- [OpenVQA](https://github.com/MILVLG/openvqa).
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [Detectron2](https://github.com/facebookresearch/detectron2)

