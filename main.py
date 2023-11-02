import argparse
import json
import os
from datetime import datetime

import librosa
from tqdm import tqdm

from analyzed_audio import AnalyzedAudio
from audio_classifier import Yamnet
from audio_enhancer import AudioEnhancer
from clap_model import ClapClassifier
from noise_reducer import NoiseReducer
from output_ranker import Ranker
from pipeline import Pipeline
from sentiment_analyzer import TextualSentimentAnalyzer
from speaker_diarization import SpeakerDiarization
from toxic_words_detection import ToxicWordsDetector
from transcriber import WhisperTranscriber
import soundfile as sf


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
    pipeline_.add_step(TextualSentimentAnalyzer(name='text_sentiment_analyzer'))
    pipeline_.add_step(Yamnet(name='audio_classifier'))
    pipeline_.add_step(SpeakerDiarization(name='speaker_diarization'))
    pipeline_.add_step(ClapClassifier(name='clap_classifier'))
    pipeline_.add_step(Ranker(name='ranker'))
    return pipeline_


def create_dir_if_not_exists(dirname='resources'):
    output_path = os.path.join("resources/runs", dirname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path


def check_audio_format(f, raw_audio_path):
    if f.subtype == 'IMA_ADPCM':
        print('bad format - IMA_ADPCM')
        data = f.read(dtype='int16')
        new_name = raw_audio_path.replace('.wav', '_fixed_format.wav').replace('.WAV', '_fixed_format.wav')
        sf.write(new_name, data, f.samplerate, subtype='PCM_16')
        raw_audio_path = new_name
    return raw_audio_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_audio_path', type=str, help='the path to the raw audio file')
    parser.add_argument('metadata', type=str, help='json object with metadata about the audio file')
    args = parser.parse_args()

    output_path = create_dir_if_not_exists()
    pipeline = init_pipeline()
    f = sf.SoundFile(args.raw_audio_path)
    raw_audio_path = check_audio_format(f, args.raw_audio_path)
    # Load the audio files
    raw_audio, sample_rate = load_audio_file(raw_audio_path)
    metadata = load_json_file(args.metadata)

    chunk_num_seconds = 60 * 5
    # split audio to chunks of 'chunk_num_seconds' seconds
    chunks = []
    for i in range(0, len(raw_audio), int(sample_rate * chunk_num_seconds)):
        chunks.append(raw_audio[i:i + sample_rate * chunk_num_seconds])

    # save chunks to files
    for i in tqdm(range(len(chunks))):
        file_name = args.raw_audio_path.split('/')[-1]
        file_name_without_extension = file_name.replace('.wav', '').replace('.WAV', '')
        new_file_name = f'{file_name_without_extension}_{i}.wav'
        chunk_output_path = os.path.join(output_path, new_file_name)

        sf.write(chunk_output_path, chunks[i], int(sample_rate))

        # run pipeline on each chunk
        input_audio = AnalyzedAudio(path=chunk_output_path,
                                    output_path=output_path, audio=chunks[i],
                                    sr=int(sample_rate), metadata=metadata)
        output_audio = pipeline(input_audio)
        chunks[i] = output_audio

    return chunks


if __name__ == '__main__':
    analyzed_audio = main()
