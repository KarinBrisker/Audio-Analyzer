import json

from analyzed_audio import AnalyzedAudio
from pipeline import Pipe


class ToxicWordsDetector(Pipe):
    def __init__(self, name):
        super().__init__(name=name)
        self.toxic_words = json.load(open("resources/toxic.json", "r"))

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        mentions = []
        transcript = analyzed.transcript
        lines = transcript.split("\n")

        for line in lines:
            transcript_words = line.split()
            transcript_words = [word.strip(".,!?") for word in transcript_words]
            i = 0

            while i < len(transcript_words):
                found_problem = False
                # Check for problematic phrases
                for phrase in self.toxic_words:
                    phrase_words = phrase.split()
                    # remove punctuation from phrase
                    phrase_to_check = transcript_words[i:i + len(phrase_words)]

                    if phrase_to_check == phrase_words:
                        mentions.append(' '.join(phrase_words))
                        i += len(phrase_words)
                        found_problem = True
                        break

                if not found_problem:
                    i += 1

        analyzed.__setattr__("toxic_mentions", mentions)

        return analyzed
