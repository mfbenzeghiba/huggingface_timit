"""Provide data arguments for the training."""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    dataset_name: str = field(
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        }
    )
    train_csv: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use \
                (via the datasets library). Defaults to 'train'"
        },
    )
    valid_csv: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation data set split to use \
                (via the datasets library). Defaults to 'validation'"
        },
    )
    vocab_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the vocab file."
        },
    )
    phone_mapping_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The file to map 61 phones to 39 or 48 phones."
        },
    )
    phone_mapping_key: Optional[str] = field(
        default='61to39',
        metadata={
            "help": "The conversion key to reduce the phoneme list.\
                Possible keys ara '61to61', '61to48' and '61to39'"
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    text_column_name: str = field(
        default="phonetic",
        metadata={
            "help": "The name of the dataset column containing the text data. Defaults to 'text'"
        },
    )
    pad_token: str = field(
        default="<pad>",
        metadata={"help": "The padding token for the tokenizer"},
    )
    unk_token: str = field(
        default="<unk>",
        metadata={"help": "The unknown token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
