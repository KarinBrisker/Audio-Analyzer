import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

from analyzed_audio import AnalyzedAudio
from pipeline import Pipe


class Yamnet(Pipe):
    # https://www.tensorflow.org/tutorials/audio/transfer_learning_audio
    def __init__(self, name):
        super().__init__(name)
        self.yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        self.yamnet_model = hub.load(self.yamnet_model_handle)
        self.class_map_path = self.yamnet_model.class_map_path().numpy().decode('utf-8')
        self.class_names = list(pd.read_csv(self.class_map_path)['display_name'])

    @tf.function
    def load_wav_16k_mono(self, filename):
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(
            file_contents,
            desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        testing_wav_data = self.load_wav_16k_mono(analyzed.path)
        scores, embeddings, spectrogram = self.yamnet_model(testing_wav_data)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.math.argmax(class_scores)
        inferred_class = self.class_names[top_class]

        print(f'The main sound is: {inferred_class}')
        print(f'The embeddings shape: {embeddings.shape}')

        analyzed.__setattr__("yamnet", inferred_class)
        return analyzed
