# Baseline eAE with VQW2V front

## Abstract

This folder contains an enhanced eAE model with cycle consistency loss and VQW2V model as a stronger front, to test if the cycle consistency loss still contributes. Compared to baseline eAE model, it shows a promising improvement.

## Data

Our experiment is mainly based on <a href='http://www.aishelltech.com/aishell_3'>AISHELL-3 dataset.

During our training process, Spk1 refers to SSB0603 and Spk2 refers to SSB0057 in AISHELL-3 dataset.

## Requirements

### Basic Environment

Simply run: `pip install -r requirements.txt`

### To install fairseq and develop locally:

``` 
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

### Wav2Vec

In our implementation, we're using Wav2Vec 2.0 Base w/o finetuning which is trained on LibriSpeech.
You can download the checkpoint [wav2vec_small.pt](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt) from [pytorch/fairseq](https://github.com/pytorch/fairseq).

## Train

Generally run:

```
python train_audio.py --data_path1 PATH_TO_TRAINING_DATA --data_path2 PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL  --load_model LOAD_MODEL_PATH(optional) --wav2vec_path WAV2VEC_PATH
```

## Test
### Voice Conversion
To convert a wavfile using a trained model, generally run:
```
python test_audio.py --model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT --wav2vec_path WAV2VEC_PATH
```

You can specify any audio data as PATH_TO_INPUT. 

### Evaluation Toolkits
Four metrics are used for quantitative evaluation, including goodness of pronunciation (GOP), character error rate (CER), MOSNet score and speaker classification accuracy (SCA).

For SCA toolkit, you can refer to <a href="https://gitlab.com/lpq96/speaker_classification_base_sunine"> SCA tool link</a>. Our classification model is based on the x-vector structure with 400 background speakers from AISHELL-1 dataset plus the target speakers from the training set.
 
For MOSNet toolkit, simply run:
```
python calc_mos.py --rootdir WAVS_TO_BE_EVALUATED --pretrained_model PRETRAINED_MODEL
```

For GOP & CER evaluation, you may use Kaldi Toolkit to prepare your own toolkit.

## Notice

Notice that all the models above should strictly follow our multi-step training method described in Methodology section in the paper.
