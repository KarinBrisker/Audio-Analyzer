import argparse
import json

import librosa

from audio_analyzer import AudioAnalyzer
from output_ranker import Ranker


def load_audio_file(path):
    """Load an audio file with librosa and return the audio and sample rate."""
    audio, sr = librosa.load(path, sr=None)
    return audio, sr


def load_json_file(path):
    """Load a JSON file and return the data."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('raw_audio_path', type=str, help='the path to the raw audio file')
    parser.add_argument('clean_audio_path', type=str, help='the path to the clean audio file after the preprocess')
    parser.add_argument('video_indexer_json', type=str, help='json object created by the video indexer')
    parser.add_argument('metadata_json', type=str, help='json object with metadata about the audio file')
    args = parser.parse_args()

    # Load the audio files
    raw_audio, raw_sr = load_audio_file(args.raw_audio_path)
    clean_audio, clean_sr = load_audio_file(args.clean_audio_path)

    # If video indexer json is a path then load it
    if args.video_indexer_json.endswith('.json'):
        args.video_indexer_json = load_json_file(args.video_indexer_json)

    vi_post_processor = VideoIndexerPostProcessor()
    audio_analyzer = AudioAnalyzer(raw_audio, raw_sr, clean_audio, clean_sr)
    ranker = Ranker()

    # Clean json file
    nlp_json = vi_post_processor(args.video_indexer_json, args.metadata_json)

    # Analyze the audio and return json object
    audio_json = audio_analyzer()

    # Merge the jsons
    final_json = {**audio_json, **nlp_json}

    output = ranker.rank(final_json)

    print(output)


if __name__ == '__main__':
    main()
