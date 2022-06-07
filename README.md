# Enhanced Exemplar Autoencoder
## Title
<b>Enhanced exemplar autoencoder with cycle consistency loss in any-to-one voice conversion</b>

## Author
Weida Liang, Lantian Li, Wenqiang Du, Dong Wang

## Abstract
We propose to solve the entanglement problem of eAE by a cycle consistency loss. The basic idea is to learn a AE model that can separate content and speaker information in speech signals with cycle consistency loss, and then perform conversion by picking up content information from the source speaker and speaker information from the target speaker. Experiments conducted on the AISHELL-3 corpus showed that this new approach improved the baseline eAE consistently.

This repository presents detailed Pytorch Implementation of our work. 4 file folders stand for different models in our work. 

## Source

<a href="http://166.111.134.19:7777/liangwd/"> project page </a>

<a href="http://166.111.134.19:7777/liangwd/paper.html"> paper </a>



## Data

Our experiment is mainly based on <a href='http://www.aishelltech.com/aishell_3'>AIShell-3 dataset.</a>

## Requirements

Go to any folder, then run: `pip install -r requirements.txt`

Note that this code borrows heavily from <a href='https://github.com/dunbar12138/Audiovisual-Synthesis'>Exemplar Autoencoder</a>

## Model Description

![](https://gitlab.com/lwd17/enhanced_examplar_ae/-/raw/main/cycle.png)

Our approach is mainly based on a cycle consistency loss. Specifically, we train eAEs of multiple speakers with a shared encoder, and meanwhile encourage the speech reconstructed from any speaker-specific decoder to get a consistent latent code as the original speech when cycled back and encoded again. Our approach shows promising improvement compared to baseline eAE.

## Train

### Baseline eAE

Check baseline_code for a baseline eAE model. Generally, run:
```
python train_audio.py --data_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL --load_model LOAD_MODEL_PATH(optional)
```

You can specify any audio data as PATH_TO_TRAINING_DATA, and a small clip of audio as PATH_TO_TEST_AUDIO. 

### Cycle eAE

Check cycle_code for a enhanced eAE model, which follows the model we described above. We present a 2-speaker training code. You may add speakers as you wish. Generally, run:
```
python train_audio.py --data_path1 PATH_TO_TRAINING_DATA --data_path2 PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL --loss_content --load_model LOAD_MODEL_PATH(optional)
```

### VQW2V+eAE

As we describe in 4.6.4 in our paper, you may use a pre-trained VQW2V model as a more powerful front for training. You may check wav2vec_eae_code.

#### To install fairseq and develop locally:

``` 
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

#### Wav2Vec

In our implementation, we're using Wav2Vec 2.0 Base w/o finetuning which is trained on LibriSpeech.
You can download the checkpoint [wav2vec_small.pt](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt) from [pytorch/fairseq](https://github.com/pytorch/fairseq).

After preparing the tools above, generally run: 

```
python train_audio.py --data_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL  --load_model LOAD_MODEL_PATH(optional) --wav2vec_path WAV2VEC_PATH
```

### VQW2V+Cycle eAE
Check wav2vec_cycle_code for a Cycle eAE model with pretrained VQW2V front. Generally follow the steps in VQW2V+eAE section, and run:
```
python train_audio.py --data_path1 PATH_TO_TRAINING_DATA --data_path2 PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL  --load_model LOAD_MODEL_PATH(optional) --wav2vec_path WAV2VEC_PATH
```

## Test
### Voice Conversion
To convert a wavfile using a trained model, generally run:
```
python test_audio.py --model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT --wav2vec_path WAV2VEC_PATH(if needed)
```

You can specify any audio data as PATH_TO_INPUT. 

### Evaluation Toolkits
Four metrics are used for quantitative evaluation, including goodness of pronunciation (GOP), character error rate (CER), MOSNet score and speaker classification accuracy (SCA).

For SCA toolkit, you can refer to <a href="https://gitlab.com/lpq96/speaker_classification_base_sunine"> SCA tool link</a>. Our classification model is based on the x-vector structure with 400 background speakers from AISHELL-1 dataset plus the target speakers from the training set.
 
For MOSNet toolkit, simply run:
```
python calc_mos.py --rootdir WAVS_TO_BE_EVALUATED --pretrained_model PRETRAINED_MODEL(optional)
```

For GOP & CER evaluation, you may use Kaldi Toolkit to prepare your own toolkit.

## Notice
Notice that all the models above should strictly follow our multi-step training method described in Methodology section in the paper.


For more details, you can refer to our paper and project page.
