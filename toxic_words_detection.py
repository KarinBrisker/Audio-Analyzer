from pipeline import Pipe


class ToxicWordsDetector(Pipe):
    def __init__(self, name):
        super().__init__(name=name)
        self.toxic_words = open("resources/toxic.json", "r").read()

    def __call__(self, analyzed):
        toxic_mentions = []
        transcript = analyzed.transcript
        for segment in transcript:
            text = segment['text']
            for word in text.split():
                if word in self.toxic_words:
                    toxic_mentions.append(word)
        analyzed.__setattr__("toxic_words", toxic_mentions)
        return analyzed
