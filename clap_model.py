import json

import torch

from analyzed_audio import AnalyzedAudio
from pipeline import Pipe
from transformers import ClapAudioModel, ClapTextModel, ClapProcessor, AutoProcessor
from transformers import AutoProcessor, ClapModel


class ClapClassifier(Pipe):
    def __init__(self, name):
        super().__init__(name=name)
        self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        # from json file
        self.classes = json.load(open("resources/clap_classes.json", "r"))


    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        inputs = self.processor(text=self.classes, audios=analyzed.audio,
                                return_tensors="pt", padding=True, sampling_rate=analyzed.sr)

        outputs = self.model(**inputs)
        logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
        probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
        _, topk_indices = torch.topk(probs, 10, dim=-1)
        topk_indices = topk_indices.tolist()[0]
        topk_class_names = [self.classes[i] for i in topk_indices]

        analyzed.__setattr__("similarity_scores", topk_class_names)
        return analyzed
