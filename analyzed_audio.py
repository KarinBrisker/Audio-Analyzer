import os
from pathlib import Path

import numpy as np


class AnalyzedAudio:
    def __init__(self, path: str, output_path, audio: np.ndarray, sr: int, metadata: dict = None):
        self.path = Path(path)
        self.output_path = output_path
        self.base_path = Path(os.path.dirname(path))
        self.filename = os.path.basename(path).strip(".wav")
        self.metadata = metadata
        self.audio = audio
        self.sr = sr
        self.clean_audio = None
        self.enhanced_audio = None
        self.sentiment = None
        self.day_part = None
        self.background_noise = None
        self.diarization = None
        self.transcript = None
        self.ranked = None
        self.toxic_words = None
        self.similarity_scores = None
        self.yamnet = None

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def to_dict(self):
        return self.__dict__

    def __str__(self):
        return str(self.__dict__)
