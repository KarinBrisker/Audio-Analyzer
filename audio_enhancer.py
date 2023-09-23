import os

from pydub import AudioSegment

from analyzed_audio import AnalyzedAudio
from pipeline import Pipe


class AudioEnhancer(Pipe):
    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        sound = AudioSegment.from_file(analyzed.path)
        louder_sound = sound + 10
        analyzed.__setattr__("enhanced_audio", louder_sound)
        output_file = os.path.join(analyzed.output_path, analyzed.filename + "_enhanced.wav")
        # Export the adjusted audio to the output file
        louder_sound.export(output_file, format="wav")
        return analyzed

