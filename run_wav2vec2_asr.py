"""Main script to train the model..."""
#!/usr/bin/env python3
import os
import sys
import logging
import json
from typing import Dict, List, Union
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import datasets
import torch

from transformers import (
    HfArgumentParser,
    Trainer,
    Wav2Vec2Config,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainerCallback,
    set_seed
)

#from hugginFace.Timit.data_processing.process_data import create_dataset, read_phone_mapping, create_vocab
from data_processing.process_data import create_dataset, read_phone_mapping, create_vocab
from metric_utils import compute_token_errors
from args.model_arguments import ModelArguments
from args.data_arguments import DataTrainingArguments


logger = logging.getLogger(__name__)

class EvalTrainCallback(TrainerCallback):
    """Evaluate the model on the train data.
    from: https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/7   
    """

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset,
                                   metric_key_prefix='train')
            return control_copy


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (
                :obj:`bool`, :obj:`str` or
                :class:`~transformers.tokenization_utils_base.PaddingStrategy`,
                `optional`, defaults to :obj:`True`
            ):
            Select a strategy to pad the returned sequences (according to the model's padding
            side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: 
                Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
            * :obj:`'max_length'`:
                Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a
                batch with sequences of different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int],
                 torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info('Training/evaluation parameters %s', training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # create the output dir if it does not exists
    modeling_unit = model_args.modeling_unit
    logger.info('The modeling unit is %s:', modeling_unit)
    output_dir = training_args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger.info('Training results will be saved in %s:', output_dir)
    logger.info('Load the dataset and apply phone mapping if needed...')
    phone_mapping = None
    if modeling_unit == 'phoneme':
        if data_args.phone_mapping_file is not None:
            phone_mappings = read_phone_mapping(data_args.phone_mapping_file)
            phone_mapping = phone_mappings[data_args.phone_mapping_key]
        else:
            logging.error('The phone mapping file is missing....')

    train_dataset = create_dataset(
        data_args.train_csv, data_args.audio_column_name,
        modeling_unit, phone_mapping=phone_mapping
    )
    if data_args.audio_column_name not in train_dataset.column_names:
        logger.error(
            '--audio_column_name %s not found in train dataset.', data_args.audio_column_name
        )
    if data_args.text_column_name not in train_dataset.column_names:
        logger.error(
            '--text_column_name %s not found in train dataset.', data_args.text_column_name
        )
    # valid dataset
    if training_args.do_eval:
        valid_dataset = create_dataset(
            data_args.valid_csv, data_args.audio_column_name,
            modeling_unit, phone_mapping=phone_mapping
        )
        if data_args.audio_column_name not in valid_dataset.column_names:
            logger.error(
                '--audio_column_name %s not found in valid dataset.', data_args.audio_column_name
            )
        if data_args.text_column_name not in valid_dataset.column_names:
            logger.error(
                '--text_column_name %s not found in valid dataset.', data_args.text_column_name
            )

    # save special tokens for tokenizer
    unk_token = data_args.unk_token
    pad_token = data_args.pad_token
    if modeling_unit == 'char':
        word_delimiter_token = data_args.word_delimiter_token

    logger.info('Create the tokenizer...')
    if data_args.vocab_file is None:
        logger.warning(
            'The vocab file is missing...Generating a vocab from train/valid dataset'
        )
        vocab_file = os.path.join(output_dir, 'vocab.json')
        vocab_dataset = train_dataset
        if training_args.do_eval:
            vocab_dataset = datasets.concatenate_datasets([train_dataset, valid_dataset])
        vocab_dict = create_vocab(vocab_dataset, modeling_unit, data_args.text_column_name)

        vocab_dict[unk_token] = len(vocab_dict)
        vocab_dict[pad_token] = len(vocab_dict)
        if modeling_unit == 'char':
            vocab_dict[word_delimiter_token] = len(vocab_dict)

        with open(vocab_file, 'w', encoding='utf-8') as vf:
            json.dump(vocab_dict, vf)
    else:
        vocab_file = data_args.vocab_file

    # add special tokens
    tokenizer_kwargs = {
        "pad_token": pad_token,
        "unk_token": unk_token
    }
    if modeling_unit == 'char':
        tokenizer_kwargs['word_delimiter_toke'] = word_delimiter_token

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file,
        **tokenizer_kwargs,
    )

    v = tokenizer.get_vocab()
    print(v)

    logger.info('Create the feature extractor...')
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )

    logger.info('Override the model config...')
    config = Wav2Vec2Config.from_pretrained(
        model_args.model_name_or_path,
    )
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
            "do_stable_layer_norm": True,  # uses pre-norm instead of norm after attention
            "ctc_zero_infinity": True,  # set to 0 the loss in case it was 'infinity' at some point
        }
    )

    logger.info('Create the model...')
    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
    )

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    logger.info('Save everything to be able to create a single processor later.')
    feature_extractor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    #processor = Wav2Vec2Processor.from_pretrained(output_dir)

    def prepare_dataset(batch):
        audio = batch[data_args.audio_column_name]
        # batched output is "un-batched"
        inputs = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"])
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = tokenizer(batch[data_args.text_column_name]).input_ids
        return batch

    train_dataset = train_dataset.map(
        prepare_dataset,
        num_proc=data_args.preprocessing_num_workers
    )

    valid_dataset = valid_dataset.map(
        prepare_dataset,
        num_proc=data_args.preprocessing_num_workers
    )

    if modeling_unit == 'phoneme':
        train_dataset = train_dataset.remove_columns(['text'])
    else:
        train_dataset = train_dataset.remove_columns(['phonetic'])

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, spaces_between_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        # adding the 'spaces_between_special_tokens' is essential with phoneme modeling
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False,
                                               spaces_between_special_tokens=True)

        nb_samples = pred_logits.shape[0]
        errors = compute_token_errors(label_str, pred_str)
        for editop, _ in errors.items():
            errors[editop] /= nb_samples
        return {
            'ins': round(errors['insert'], 3),
            'sub': round(errors['replace'], 3),
            'del': round(errors['delete'], 3),
            'ter': round(errors['token_errors'], 3)
        }

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        tokenizer=feature_extractor,
    )
    trainer.add_callback(EvalTrainCallback(trainer))

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_results = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

    # Evaluation
    logger.info('Evaluate the model...')
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval",eval_metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()
