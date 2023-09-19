import torch

from analyzed_audio import AnalyzedAudio
from pipeline import Pipe
from transformers import ClapAudioModel, ClapTextModel, ClapProcessor


class ClapClassifier(Pipe):
    def __init__(self, name):
        super().__init__(name=name)
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
        self.audio_model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
        self.text_model = ClapTextModel.from_pretrained("laion/clap-htsat-fused")
        self.classes = ["violence", "screaming", "gunshot", "explosion", "car", "glass", "music", "animal", "human",
                        "nature", "other", "crying", "water", "silence", "applause", "laughter", "whispering",
                        "clapping", "anger", "falling", "footsteps", "door", "knocking", "phone", "alarm", "bell",
                        "siren", "horn", "engine", "baby", "speech", "fear"]

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        # Process the audio file and text description
        inputs_audio = self.processor(analyzed.path, sampling_rate=16000, return_tensors="pt", padding=True)

        # Encode the audio file
        with torch.no_grad():
            embeddings_audio = self.audio_model(**inputs_audio).last_hidden_state

        similarity_scores = dict.fromkeys(self.classes)

        # Process each text description, encode it, and compute its similarity with the audio segment
        for sound_class in self.classes:
            inputs_text = self.processor(sound_class, return_tensors="pt", padding=True)
            with torch.no_grad():
                embeddings_text = self.text_model(**inputs_text).last_hidden_state
            similarity = torch.nn.functional.cosine_similarity(embeddings_audio, embeddings_text)
            similarity_scores[sound_class] = similarity.item()

        analyzed.__setattr__("similarity_scores", similarity_scores)
        return analyzed
