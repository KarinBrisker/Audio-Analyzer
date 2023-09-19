import os

from pydub import AudioSegment
import soundfile as sf

from analyzed_audio import AnalyzedAudio
from pipeline import Pipe


class AudioEnhancer(Pipe):
    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        sound = AudioSegment.from_file(analyzed.path)
        louder_sound = sound + 10
        analyzed.__setattr__("enhanced_audio", louder_sound)
        output_file = os.path.join(analyzed.base_path, analyzed.filename + "_enhanced.wav")
        # Save the clean audio to a WAV file
        sf.write(output_file, louder_sound, analyzed.sr, subtype='PCM_16')
        return analyzed

