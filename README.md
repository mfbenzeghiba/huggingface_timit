# Fine-tuning Wav2vec2 on Timit
## Introduction
This repo fine-tunes the wav2vec2.0 pretrained model on TIMIT using hugging face modules.
There are numerous works on this task, but the difference in the performances among these works,
suggests that something went wrong with some implementations, ans decided to do it myself.

The fine-tuning is done for both phoneme and character recognition tasks using CTC.

## Data processing
The data processing is performed with the script `data_processing/prepare_data.py`.
It creates three different csv files (train, dev and test). Each file consists of three columns.
1. The *audio* column contains the path to the audio file.
2. The *phonemes_transcription* column contains the corresponding phone transcription
3. The *text_transcription* column contains  the correspoding text for character recongition.

Train, valid and test files contain `3692`, `400` and `192` files respectively.

## Training

Training is performed with the `finetune_timit.sh`. This script define the training arguments and the model parameters,
and launch the main training script `run_wav2vec2_timit.py`.

> [!NOTE]
  Although the TIMIT dataset was released with a set of 61 phonemes, in practice, this set is often reduced to 48 or 39 phonemes.
Wich set the original 61 phonemes will be reduced to is specified with the `phone_mapping_key` argument.

> [!IMPORTANT]
   The best model is selected according to the phoneme/character edit distance computed on 
the validation set. Since some phonemes are composed of two characters, **to compute the edit distance correctly,
it is _extremly_ important to set the argument `spaces_between_special_tokens=True`** in the batch_decode method.

## Evaluation

Evaluation of the best model is performed with the `eval_model.py` script.
This will run the evaluation of the model on the eval file for a phonemes recognition task with a reduced set of 39 phonemes.
```
eval_model.py
  --model_path <best_checkpoint>
  --phone_mapping_file phones.60-48-39.map.txt
  --eval_file <the processed csv file>
  --phone_mapping_key '61to39'
  --modeling_unit phoneme
  --audio_column_name 'audio'
  --text_column_name 'phonetic'
```

When the model is evaluated on the character recognition task, the `--phone_mapping_key` and the `--phone_mapping_file` arguments are not needed.
Table bellow reports the results in terms of CER (Character Error Rate) and PER (Phoneme Error Rate) on valid and test datasets.

|       | PER (%)  |
|:------|--------:|
| Valid | **15.0** |
| Test  | **16.5** |

