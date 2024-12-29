"""Functions to process the data."""

import os
import logging
from typing import Dict, List
from functools import reduce
import regex

import pandas as pd
import datasets


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def read_phone_mapping(phone_mapping_file: str) -> Dict:
    """Read the phone mapping for timit.

    Args:
        phone_mapping_file (str): The phone mapping file.

    Returns:
        dict: The mapping dictionary
    """

    pm = dict()
    with open(phone_mapping_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [l.strip().split() for l in lines]
    pm['61to61'] = {line[0]: line[0] for line in lines if len(line) == 3}
    pm['61to48'] = {line[0]: line[1] for line in lines if len(line) == 3}
    pm['48to39'] = {line[1]: line[2] for line in lines if len(line) == 3}
    pm['61to39'] = {line[0]: line[2] for line in lines if len(line) == 3}

    return pm


def read_data_file(csv_file: str) -> Dict:
    """Read the data csv file.

    Args:
        csv_file (str): Path to the csv file

    Returns:
        dict: dictionary that contains the necessary information to train a model.
    """

    audio_files = []
    phonemes_transcriptions = []
    text_transcriptions = []
    df_data = pd.read_csv(csv_file)

    for _, row in df_data.iterrows():
        audio_files.append(row['audio'])
        y = row['phonemes_transcription']
        y = y.split('\'')[1]
        phonemes_transcriptions.append(y)
        z = row['text_transcription'][2:-2]
        text_transcriptions.append(z)

    return datasets.Dataset.from_dict(
        {
            'audio_file': audio_files,
            'phonetic': phonemes_transcriptions,
            'text': text_transcriptions
        }
    )


def remove_special_characters(item: str) -> str:
    """Remove special characters.

    Args:
        item (str): sentence

    Returns:
        str: The sentence after removing the special character.
    """
    item['text'] = regex.sub('[?,.\\\\!;:"-]', '', item['text']).lower()
    return item


def reduce_phoneset(item: str, phone_mapping: Dict[str, str] = None) -> str:
    """Reduce phone set.

    Args:
        item (str): sequence of phones
        phone_mapping (dict): mapping dictionary to reduce the phoneme set.
    """

    if phone_mapping is not None:
        sentence = item['phonetic'].split()
        item['phonetic'] = ' '.join([phone_mapping[x] for x in sentence])

    return item


def create_dataset(csv_file: str, audio_column_name: str, modeling_unit: str='char',
                   phone_mapping=None) -> datasets:
    """Create the Dataset for the Timit database.

    Args:
        csv_file (str): The csv file.
        audio_column_name (str): the name of audio column
        modeling_unit (str, optional): The modeling unit, either char or phoneme.
        phone_mapping (dict, optional): The dictionary to map 61 phones to either 39 or 48 phones
        in case we reduce the phoneset..

    Returns:
        datasets: _description_
    """

    dataset = read_data_file(csv_file)
    if modeling_unit == 'char':
        dataset = dataset.map(remove_special_characters)
    else:
        dataset = dataset.map(reduce_phoneset, fn_kwargs={"phone_mapping": phone_mapping})
    dataset = (
        dataset.cast_column("audio_file", datasets.Audio(sampling_rate=16_000))
        .rename_column('audio_file', audio_column_name)
    )
    return dataset


def create_vocab(dataset, modeling_unit: str, text_column_name: str) -> Dict[str, int]:
    """Create the vocab.

    Args:
        dataset (_type_): The dataset.
        modeling_unit (str): The modeling unit, 'phoneme' or 'char.
        text_column_name (str): The name of the column.

    Returns:
        dict: The phoneme/char vocabulary 
    """

    if modeling_unit == 'char':
        word_list = [set(x[text_column_name]) for x in dataset]
        vocab_list = list(reduce(lambda x, y: x.union(y), word_list))
    else:
        vocab_list = list(set([token for x in dataset for token in x[text_column_name].split()]))

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    if modeling_unit == 'char':
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
    return vocab_dict


def dump_train_results(output_dir: str, train_results: List) -> None:
    """Dump the training results to a log file.

    Args:
        output_dir (str): The output dir to create the log file.
        train_results (List): The training results (the state.log_history)
    """

    log_file = os.path.join(output_dir, 'training.log')
    train_keys_to_keep = ['epoch', 'train_runtime', 'train_loss',
                          'train_ins', 'train_sub', 'train_del','train_cer']
    eval_keys_to_keep = ['epoch', 'eval_loss', 'eval_ins', 'eval_sub', 'eval_del', 'eval_cer']

    train_res = [item for item in train_results if 'train_cer' in item]
    eval_res = [item for item in train_results if 'eval_cer' in item]

    train_df = pd.DataFrame(train_res, columns=train_keys_to_keep)
    eval_df = pd.DataFrame(eval_res, columns=eval_keys_to_keep)
    df = pd.merge(train_df, eval_df, on=['epoch'])

    df.to_csv(log_file, index=False)
