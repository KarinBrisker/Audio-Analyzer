from abc import ABCMeta, abstractmethod
from typing import List

from analyzed_audio import AnalyzedAudio


class Pipeline:
    def __init__(self, pipe_name, steps=None):
        self.pipe_name: str = pipe_name
        self.steps: List[Pipe] = steps if steps else []

    def add_step(self, step):
        self.steps.append(step)

    def __call__(self, updated_audio: AnalyzedAudio):
        for step in self.steps:
            updated_audio = step(updated_audio)
        return updated_audio


class Pipe(metaclass=ABCMeta):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        pass

