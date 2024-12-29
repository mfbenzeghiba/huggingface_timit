"""Provide model arguments."""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded"
                    "from huggingface.co"
        }
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={
            "help": "Whether to freeze the feature encoder layers of the model."
        }
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "The dropout ratio for the attention probabilities."
        }
    )
    activation_dropout: float = field(
        default=0.1,
        metadata={
            "help": "The dropout ratio for activations inside the fully connected layer."
        }
    )
    feat_proj_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout ratio for the projected features."
        }
    )
    hidden_dropout: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings,"
                    "encoder, and pooler."
        }
    )
    final_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for the final projection layer."
        }
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen"
                " as the start of the vector span to be masked. Approximately"
                " 'mask_time_prob * sequence_length // mask_time_length' feature"
                " vectors will be masked along the time axis."
            )
        }
    )
    mask_time_length: int = field(
        default=10,
        metadata={
            "help": "Length of vector span to mask along the time axis."
        }
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the feature axis to be chosen as"
                " the start of the vectorspan to be masked. Approximately"
                " 'mask_feature_prob * sequence_length // mask_feature_length' feature"
                " bins will be masked along the time axis."
            )
        }
    )
    mask_feature_length: int = field(
        default=10,
        metadata={
            "help": "Length of vector span to mask along the feature axis."
        }
    )
    layerdrop: float = field(
        default=0.0,
        metadata={
            "help": "The LayerDrop probability."
        }
    )
    ctc_loss_reduction: Optional[str] = field(
        default="mean",
        metadata={
            "help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."
        }
    )
    modeling_unit: Optional[str] = field(
        default="char",
        metadata={
            "help": "The modeling unit. sould be 'char' or 'phoneme'."
        }
    )
    eval_on_start: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to perform a evaluation step (sanity check) before the training\
                    to ensure the validation steps works correctly."
        }
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to log verbose messages or not."
        }
    )
