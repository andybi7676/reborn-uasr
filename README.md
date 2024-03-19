# REBORN: Reinforcement-Learned Boundary Segmentation with Iterative Training for Unsupervised ASR 

Liang-Hsuan Tseng, En-Pei Hu, Cheng-Han Chiang, Yuan Tseng, Hung-yi Lee, Lin-shan Lee, Shao-Hua Sun
#### National Taiwan University
[![arXiv](https://img.shields.io/badge/arXiv-Paper-color.svg)](https://arxiv.org/abs/2402.03988) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andybi7676/reborn-uasr/blob/main/hf/reborn_demo_colab.ipynb) [![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collections-orange)](https://huggingface.co/spaces/Audio-AGI/AudioSep) [![Docker Hub](https://img.shields.io/badge/Docker%20Hub-Image-3385ff.svg)](https://hub.docker.com/r/andybi7676/reborn-uasr)

This repository is dedicated to the "reborn-uasr" project, an initiative focused on enhancing Unsupervised Automatic Speech Recognition (ASR) through the implementation of Reinforcement Learning (RL) techniques for segmenter training.

## Using REBORN Models through Hugging Face 🤗

The simplest way to access the REBORN models is through Hugging Face. We have wrapped our model including PCA dimension reduction matrix, REBORN segmenter, and REBORN generator into the Hugging Face supported form. Furthermore, we've also built the datasets corresponding to the models to Hugging Face (LibrSpeech 100 hours, Multilingual LibriSpeech across 6 languages). For those who want to have a quick start, please checkout our [demo on Google Colab](https://colab.research.google.com/github/andybi7676/reborn-uasr/blob/main/hf/reborn_demo_colab.ipynb). 

### Summarizing Card Names

To replicate the REBORN end-to-end unsupervised phoneme recognition result, one would need:
* The upstream model (wav2vec 2.0) as feature extracter.
* The REBORN model (including the PCA dimension reduction matrix, the segmenter, and the generator).
* The corresponding dataset.

Since each of the components are all available on Hugging Face, we can follow the [demo on Google Colab](https://colab.research.google.com/github/andybi7676/reborn-uasr/blob/main/hf/reborn_demo_colab.ipynb) and generate the results across different datasets by simply replace the model card name and the dataset name. We summarize the pairing model name and dataset below for simplier usage:
| Description       | upstream_model_card | reborn_model_card | dataset_card | dataset_name |    split    |
|:------------------|:-------------------:|:-----------------:|:------------:|:------------:|:-----------:|
|LibriSpeech 100 hour @ iter2-stage1|facebook/wav2vec2-large-lv60|andybi7676/reborn-uasr_ls100h_iter2-stage1|andybi7676/reborn-uasr_librispeech-no-silence-100hr||{train.clean.100, dev.clean, dev.other, test.clean, test.other, dev.clean.small}|
|LibriSpeech 100 hour @ iter5-stage1|facebook/wav2vec2-large-lv60|andybi7676/reborn-uasr_ls100h_iter5-stage1|andybi7676/reborn-uasr_librispeech-no-silence-100hr||{train.clean.100, dev.clean, dev.other, test.clean, test.other, dev.clean.small}|
|Multilingual LibriSpeech 100 hour German @ iter2-stage1|facebook/wav2vec2-large-xlsr-53|andybi7676/reborn-uasr_mls-de_iter2-stage1|andybi7676/reborn-uasr_multilingual-librispeech-no-silence-100hr|german|{train.100hr, dev, test, dev.small}|
|Multilingual LibriSpeech 100 hour Dutch @ iter2-stage1|facebook/wav2vec2-large-xlsr-53|andybi7676/reborn-uasr_mls-de_iter2-stage1|andybi7676/reborn-uasr_multilingual-librispeech-no-silence-100hr|dutch|{train.100hr, dev, test, dev.small}|
|Multilingual LibriSpeech 100 hour French @ iter2-stage1|facebook/wav2vec2-large-xlsr-53|andybi7676/reborn-uasr_mls-de_iter2-stage1|andybi7676/reborn-uasr_multilingual-librispeech-no-silence-100hr|french|{train.100hr, dev, test, dev.small}|
|Multilingual LibriSpeech 100 hour Spanish @ iter2-stage1|facebook/wav2vec2-large-xlsr-53|andybi7676/reborn-uasr_mls-de_iter2-stage1|andybi7676/reborn-uasr_multilingual-librispeech-no-silence-100hr|spanish|{train.100hr, dev, test, dev.small}|
|Multilingual LibriSpeech 100 hour Italian @ iter2-stage1|facebook/wav2vec2-large-xlsr-53|andybi7676/reborn-uasr_mls-de_iter2-stage1|andybi7676/reborn-uasr_multilingual-librispeech-no-silence-100hr|italian|{train.100hr, dev, test, dev.small}|
|Multilingual LibriSpeech 100 hour Portuguese @ iter2-stage1|facebook/wav2vec2-large-xlsr-53|andybi7676/reborn-uasr_mls-de_iter2-stage1|andybi7676/reborn-uasr_multilingual-librispeech-no-silence-100hr|portuguese|{train.100hr, dev, test, dev.small}|

## Prerequisite
If you want to build up the environment and train the REBORN model by your own, please follow the below content to meet the requirements. 

### Docker Image
We provide the pre-built docker image on the [Docker Hub](https://hub.docker.com/r/andybi7676/reborn-uasr). The image contains all the dependencies for training reborn. This might be the simpliest way to setup the environment if you are familiar with Docker. Type the following command to pull and run the container based on the image.

`docker run -it --gpus all andybi7676/reborn-uasr:latest`

Note that this is just an example of using the image in interactive mode with all the gpus. Feel free to use it on your own way If the gpus are not available inside the container, please verify that [nvidia-docker](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu) is installed.

### Building up the Environment by Yourself
In this section we are going to give instructions on how to build up the REBORN environment step by step.
#### Fairseq
We have attach the fairseq version we use in the folder `reborn-uasr/fairseq`. You can use it by cloning our repo to make sure that there is no version biases which may possibly lead to unexpected errors. 
```shell
git clone https://github.com/andybi7676/reborn-uasr.git
cd reborn-uasr/fairseq
pip install -e .
```
#### Kenlm
Please follow the instruction from [the official repo of kenlm](https://github.com/kpu/kenlm). Please make sure that the python bindings is also installed (`pip install https://github.com/kpu/kenlm/archive/master.zip`).

#### Other requirements (python packages)
```shell
cd /your/path/to/reborn-uasr
pip install -r requirements.txt
```
Modify and run `path.sh` to export fairseq and reborn-uasr to PYTHONPATH. 
1. Modify the /path/to/fairseq to export the corrent fairseq path into the environment. 
2. run `source path.sh` to append `fairseq` and `reborn-uasr` into the PYTHONPATH. The results should be as follows:
   ```
   (base) username@desktop:/your/path/to/reborn-uasr$ source path.sh 
   Added /your/path/to/fairseq to PYTHONPATH
   Appended /your/path/to/reborn-uasr to PYTHONPATH
   =======================================================================================
   FAIRSEQ_ROOT: /your/path/to/fairseq
   REBORN_WORK_DIR: /your/path/to/reborn-uasr
   PYTHONPATH: /your/path/to/fairseq:/your/path/to/reborn-uasr
   Please make sure that FAIRSEQ_ROOT and REBORN_WORK_DIR are in PYTHONPATH
   During each runtime, please make sure to run `source path.sh` to set up the environment.
   =======================================================================================
   Testing the required import functionality...
   SUCCESS
   ```

#### Flashlight python bindings (optional)
TBA
#### Pykaldi and Kaldi (optional)
TBA

## Training REBORN

In this section, we will introduce how to train your own reborn model from scratch. Before diving into the training part, we recommend users go through the [Prerequisite](##prerequisite) section and make sure that all the requirements have been satisfied. 

We divide the training process as three main stage: [wav2vec-U initialization](###stage-0), [segmenter training](###stage-1), and [generator (phoneme prediction model) training](###stage-2). 

### Data Preparation
#### Audio preparation
#### Text preparation
### Stage 0: Training wav2vec-U as Initialization

### Stage 1: REBORN segmenter training
#### Behaviour Cloning
#### Reinforcement Learning

### Stage 2: REBORN generator training
#### Boundary post-processing
#### GAN-training

## Reference Repositories
* [fairseq](https://github.com/facebookresearch/fairseq)
* [kenlm](https://github.com/kpu/kenlm)
* [rVADfast](https://github.com/zhenghuatan/rVADfast)
* [kaldi](https://github.com/kaldi-asr/kaldi)
* [pykaldi](https://github.com/pykaldi/pykaldi)
* [flashlight](https://github.com/flashlight/flashlight)
* [transformers](https://github.com/huggingface/transformers)
* [datasets](https://github.com/huggingface/datasets)

## Citation
Please cite this work as:
```
@article{tseng2024reborn,
  title={REBORN: Reinforcement-Learned Boundary Segmentation with Iterative Training for Unsupervised ASR},
  author={Tseng, Liang-Hsuan and Hu, En-Pei and Chiang, Cheng-Han and Tseng, Yuan and Lee, Hung-yi and Lee, Lin-shan and Sun, Shao-Hua},
  journal={arXiv preprint arXiv:2402.03988},
  year={2024}
}
```

<!-- ## Setup and Use of Wav2Vec-U Models
### Installation and Configuration
1. **Install Fairseq**: Initially, use your current version of fairseq. If incompatibility issues arise, switch to the version specified in this repository.
2. **Environment Configuration**: Create an `env.yaml` file in the `$WORK_DIR/uasr-rl` directory. This file, which is ignored by git, can store custom variables for development purposes.
3. **Workspace Directory Configuration**: In your `$WORK_DIR/uasr-rl/env.yaml`, set the `WORK_DIR` variable as follows:
   ```
   WORK_DIR: /home/username/uasr-rl
   ```
4. **Model Loading Test**: Verify the model setup by running `load_w2vu_example.py`.
   ```
   cd $WORK_DIR/uasr-rl/rl/utils
   python load_w2vu_example.py
   ```

### Alternative UASR Model Loading Method
For a streamlined approach to loading the UASR model without Fairseq, use the script in `rl/transformer_behavior_cloning/load_w2vu_no_fairseq_example.py`.

## Dataset Setup
* **LibriSpeech ASR Corpus**: Available at [OpenSLR](https://www.openslr.org/12).
* **Data Preprocessing**: Follow the [wav2vec-U](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/README.md) guide for PCA preprocessed data.

## Reinforcement Learning Training
Detailed instructions for RL training of the CNN segmenter are provided in the [rl/cnn_segmenter/](rl/cnn_segmenter/) folder.

## Phoneme Segmentation F1 Evaluation
For an in-depth understanding of our phoneme segmentation F1 score evaluation methodology, please refer to the script located at [phoneseg_eval.py](s2p/scripts/phoneseg_eval.py). -->