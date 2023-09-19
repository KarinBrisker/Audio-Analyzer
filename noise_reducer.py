import os.path

import noisereduce as nr
import soundfile as sf

from analyzed_audio import AnalyzedAudio
from pipeline import Pipe


class NoiseReducer(Pipe):
    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        clean_audio = nr.reduce_noise(y=analyzed.audio, sr=analyzed.sr, stationary=False)
        analyzed.__setattr__("clean_audio", clean_audio)

        output_file = os.path.join(analyzed.base_path, analyzed.filename + "_cleaner.wav")
        # Save the clean audio to a WAV file
        sf.write(output_file, clean_audio, analyzed.sr, subtype='PCM_16')

        return analyzed
