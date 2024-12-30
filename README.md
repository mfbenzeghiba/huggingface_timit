# Fine-tuning Wav2vec2 on Timit
## Introduction
This repo fine-tunes the wav2vec2.0 pretrained model on TIMIT using hugging face modules.
There are numerous works on this task, but the difference in the performances among these works,
suggests that something went wrong with some implementations, ans decided to do it myself.

The fine-tuning is done for both phoneme and character recognition tasks using CTC.

## Data processing
The data processing is performed with the script **data_processing/prepare_data.py**.
It creates three different csv files (train, dev and test). Each file consists of three columns.
1. The *audio* column contains the path to the audio file.
2. The *phonemes_transcription* column contains the corresponding phone transcription
3. The *text_transcription* column contains  the correspoding text for character recongition.

## Training

Training is performed with the **finetune_timit.sh**. This script define the training arguments and the model parameters,
and launch the main training script **run_wav2vec2_asr**.

1. Although the TIMIT dataset was released with a set of 61 phonems, In practice, this set is often reduced to 48 or 39 phonemes.
To wich set the original 61 phonemes will be reduced to is specified with the **phone_mapping_key** argument.
2. The best model is selected according to the phoneme/character edit distance computed on 
the validation set. Since some phonemes are composed of two characters, **to compute the edit distance correctly,
it is _extremly_ important to set the argument 'spaces_between_special_tokens=True'.** in the batch_decode method.
