# Baseline Exemplar Autoencoder

## Abstract

This folder contains our baseline eAE model, which heavily borrows from <a href='https://github.com/dunbar12138/Audiovisual-Synthesis'>Exemplar Autoencoder</a>. 

## Data

Our experiment is mainly based on <a href='http://www.aishelltech.com/aishell_3'>AISHELL-3 dataset.

## Requirements

Simply run: `pip install -r requirements.txt`

## Train

Generally, run:
```
python train_audio.py --data_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL
```

## Test
### Voice Conversion
To convert a wavfile using a trained model, generally run:
```
python test_audio.py --model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT
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
