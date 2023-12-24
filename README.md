# wav2vec-URL: Boosting Performance of Unsupervised ASR with RL-Trained Segmenter
This repository is dedicated to the "wav2vec-URL" project, an initiative focused on enhancing Unsupervised Automatic Speech Recognition (ASR) through the implementation of Reinforcement Learning (RL) techniques for segmenter training.

## Setup and Use of Wav2Vec-U Models
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
For an in-depth understanding of our phoneme segmentation F1 score evaluation methodology, please refer to the script located at [phoneseg_eval.py](s2p/scripts/phoneseg_eval.py).