from analyzed_audio import AnalyzedAudio
from pipeline import Pipe


class ClapClassifier(Pipe):
    def __init__(self, name):
        super().__init__(name=name)

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        return analyzed
