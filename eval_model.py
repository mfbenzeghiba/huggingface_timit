"""Script to evaluate HF trained models."""

import os
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset, load_metric
import safetensors
import torch
import soundfile as sf
from evaluate import load


from data_processing.process_data import create_dataset, read_phone_mapping


def parse_args(args: list=None) ->  argparse.Namespace:
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
            help='For phone reduction set.')
    parser.add_argument('--modeling_unit', type=str,
            default='phoneme',
            help='The modeling unit.')
    parser.add_argument('--audio_column_name', type=str,
            default='audio',
            help='Name of the qudio column.')
    parser.add_argument('--text_column_name', type=str,
            default='phonetic',
            help='Name of the transcription column.')
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

    wer = load('wer', trust_remote_code=True)
    total_errors = 0
    
    for audio, phone_sequence in zip(eval_dataset[options.audio_column_name], eval_dataset[options.text_column_name]):
        audio_file = audio['path']
        audio_input, _ = sf.read(audio_file)
        inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=False)
        reference = phone_sequence
        with torch.no_grad():
            logits = model(inputs.input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_ids[predicted_ids == -100] = processor.tokenizer.pad_token_id
            predicted = processor.batch_decode(predicted_ids, spaces_between_special_tokens=True)[0]
            predicted = predicted.replace(' ', '')
            predicted = predicted.replace('<s>', ' ')
            wer_score = wer.compute(predictions=[predicted], references=[reference])
            total_errors += wer_score

            print("reference:", phone_sequence)
            print("predicted:", predicted)
            print('Phone Error Rate:', round(wer_score, 3))
            print("--")

    print('Total Error Rate:', round(total_errors/len(eval_dataset), 3))


if __name__ == '__main__':

    args = parse_args()
    main(args)
