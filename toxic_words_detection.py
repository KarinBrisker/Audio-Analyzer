from analyzed_audio import AnalyzedAudio
from pipeline import Pipe


class ToxicWordsDetector(Pipe):
    def __init__(self, name):
        super().__init__(name=name)
        self.toxic_words = open("resources/toxic.json", "r").read()

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        mentions = []
        lines = analyzed.transcript.split("\n")

        for line in lines:
            words = line.split()
            i = 0

            while i < len(words):
                found_problem = False
                # Check for problematic phrases
                for phrase in self.toxic_words:
                    phrase_words = phrase.split()
                    # remove punctuation from phrase
                    phrase_to_check = [word.strip(".,!?") for word in words[i:i + len(phrase_words)]]

                    if phrase_to_check == phrase_words:
                        mentions.append(' '.join(phrase_words))
                        i += len(phrase_words)
                        found_problem = True
                        break

                if not found_problem:
                    i += 1

        analyzed.__setattr__("toxic_words", mentions)

        return analyzed
