import wave

import numpy as np
import pandas as pd
import scipy
import scipy.signal
import tensorflow_hub as hub

from analyzed_audio import AnalyzedAudio
from pipeline import Pipe


class AudioClassifier(Pipe):
    # https://www.tensorflow.org/tutorials/audio/transfer_learning_audio
    def __init__(self, name):
        super().__init__(name)
        self.yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        self.yamnet_model = hub.load(self.yamnet_model_handle)
        self.class_map_path = self.yamnet_model.class_map_path().numpy().decode('utf-8')
        self.class_names = list(pd.read_csv(self.class_map_path)['display_name'])

    def load_wav(self, filename):
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        with wave.open(filename, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            wav = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
            wav = wav.astype(np.float32) / 32768  # Convert to [-1.0, 1.0]

        # Check if audio is stereo
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)  # Convert stereo to mono by averaging channels

        # Resample to 16 kHz
        if sample_rate != 16000:
            wav = scipy.signal.resample_poly(wav, up=16000, down=sample_rate)

        return wav

    @staticmethod
    def get_wav_format(filename):
        with wave.open(filename, 'rb') as wav_file:
            sample_width = wav_file.getsampwidth()
            n_channels = wav_file.getnchannels()
        return sample_width * 8, n_channels

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        # print("\nRunning Audio Classifier")
        file_path = str(analyzed.path)
        bit_depth, channels = self.get_wav_format(file_path)
        print(f'The file is {bit_depth}-bit and has {channels} channel(s).')

        wav_data = self.load_wav(str(analyzed.path))
        # Run the model, check the output.
        scores, embeddings, spectrogram = self.yamnet_model(wav_data)
        scores_np = scores.numpy()
        inferred_class = self.class_names[scores_np.mean(axis=0).argmax()]

        print(f'\nThe main sound is: {inferred_class}')
        print(f'\nThe embeddings shape: {embeddings.shape}')

        analyzed.__setattr__("yamnet", inferred_class)
        print("\nAudio Classifier finished")
        return analyzed
