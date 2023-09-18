import argparse
import json

import librosa

from analyzed_audio import AnalyzedAudio
from audio_enhancer import AudioEnhancer
from clap_model import ClapClassifier
# from audio_classifier import Yamnet
from noise_reducer import NoiseReducer
from output_ranker import Ranker
from pipeline import Pipeline
# from speaker_diarization import SpeakerDiarization
from toxic_words_detection import ToxicWordsDetector
from transcriber import WhisperTranscriber


# from sentiment_analyzer import SentimentAnalyzer


def load_audio_file(path):
    """Load an audio file with librosa and return the audio and sample rate."""
    audio, sr = librosa.load(path, sr=None)
    return audio, sr


def load_json_file(path):
    """Load a JSON file and return the data."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def init_pipeline():
    pipeline_ = Pipeline(pipe_name='audio_indexer')

    pipeline_.add_step(NoiseReducer(name='noise_reducer'))
    pipeline_.add_step(AudioEnhancer(name='audio_enhancer'))
    pipeline_.add_step(WhisperTranscriber(name='transcriber'))
    pipeline_.add_step(ToxicWordsDetector(name='toxic_words_detector'))
    # pipeline_.add_step(SentimentAnalyzer(name='text_sentiment_analyzer'))
    # pipeline_.add_step(Yamnet(name='audio_classifier'))
    # pipeline_.add_step(SpeakerDiarization(name='speaker_diarization'))
    pipeline_.add_step(ClapClassifier(name='clap_classifier'))
    pipeline_.add_step(Ranker(name='ranker'))
    return pipeline_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_audio_path', type=str, help='the path to the raw audio file')
    parser.add_argument('metadata', type=str, help='json object with metadata about the audio file')
    args = parser.parse_args()

    pipeline = init_pipeline()

    # Load the audio files
    raw_audio, sample_rate = load_audio_file(args.raw_audio_path)
    metadata = load_json_file(args.metadata)
    input_audio = AnalyzedAudio(args.raw_audio_path, raw_audio, int(sample_rate), metadata)
    output_audio = pipeline(input_audio)
    return output_audio


if __name__ == '__main__':
    analyzed_audio = main()
