"""Script to evaluate HF trained models."""

import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf

from data_processing.process_data import create_dataset, read_phone_mapping
from metric_utils import compute_token_errors


def parse_args() ->  argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description='Evaluate HF models.')
    parser.add_argument('--model_path', type=str,
            help='Path to the model location')
    parser.add_argument('--phone_mapping_file', type=str,
            help='Path to the phone mapping file.')
    parser.add_argument('--eval_file', type=str,
            help='Keep the punction in the cleaned text and convert them to a single punctuation.')
    parser.add_argument('--phone_mapping_key', type=str,
            default='61to39',
            choices=['61to61', '61to48', '61to39'],
            help='For phone reduction set.')
    parser.add_argument('--modeling_unit', type=str,
            default='phoneme',
            choices=['phoneme', 'char'],
            help='The modeling unit.')
    parser.add_argument('--audio_column_name', type=str,
            default='audio',
            help='Name of the qudio column.')
    parser.add_argument('--text_column_name', type=str,
            default='phonetic',
            choices=['phonetic','text'],
            help="Name of the transcription column. 'phonetic' for phoneme \
                 and 'text' for character.")

    args = parser.parse_args()
    return args


def main(options: dict):
    """Main function to do the evaluation.

    Args:
        options (dict): _description_
    """

    processor = Wav2Vec2Processor.from_pretrained(options.model_path)
    model = Wav2Vec2ForCTC.from_pretrained(options.model_path , use_safetensors=True)
    phone_mapping = None
    if options.modeling_unit == 'phoneme' and options.phone_mapping_file is not None:
        phone_mappings = read_phone_mapping(options.phone_mapping_file)
        phone_mapping = phone_mappings[options.phone_mapping_key]

    eval_dataset = create_dataset(options.eval_file,
                                  options.audio_column_name,
                                  options.modeling_unit,
                                  phone_mapping)

    total_errors = 0
    audio_col_name = options.audio_column_name
    text_col_name = options.text_column_name
    for audio, phone_sequence in zip(eval_dataset[audio_col_name], eval_dataset[text_col_name]):
        audio_file = audio['path']
        audio_input, _ = sf.read(audio_file)
        inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=False)
        reference = phone_sequence
        with torch.no_grad():
            logits = model(inputs.input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_ids[predicted_ids == -100] = processor.tokenizer.pad_token_id
            predicted = processor.batch_decode(predicted_ids, spaces_between_special_tokens=True)[0]
            if options.modeling_unit == 'char':
                reference = ' '.join(list(reference))
                reference = reference.replace('   ', ' <s> ')
                predicted = predicted.replace('   ', ' <s> ')
            seq_error = compute_token_errors(refs=[predicted], hyps=[reference])
            total_errors += seq_error['token_errors']

            print("reference:", reference)
            print("predicted:", predicted)
            print('Phone Error Rate:', round(seq_error['token_errors'], 3))
            print("--")

    print('Total Error Rate:', round(total_errors/len(eval_dataset), 3))


if __name__ == '__main__':

    args = parse_args()
    main(args)
