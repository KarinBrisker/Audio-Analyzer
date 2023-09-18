import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np

from analyzed_audio import AnalyzedAudio
from pipeline import Pipe


class NoiseReducer(Pipe):
    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        clean_audio = nr.reduce_noise(y=analyzed.audio, sr=analyzed.sr, stationary=False)
        analyzed.__setattr__("clean_audio", clean_audio)
        return analyzed

