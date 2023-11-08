from tqdm import tqdm

from HeBERT.src.HebEMO import HebEMO
from analyzed_audio import AnalyzedAudio
from pipeline import Pipe
from transformers import AutoTokenizer, AutoModel, pipeline


class TextualSentimentAnalyzer(Pipe):
    def __init__(self, name):
        super().__init__(name)
        self.tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_sentiment_analysis")
        self.model = AutoModel.from_pretrained("avichr/heBERT_sentiment_analysis")
        self.analyzer = pipeline(
            task="sentiment-analysis",
            model="avichr/heBERT_sentiment_analysis",
            tokenizer="avichr/heBERT_sentiment_analysis",
            top_k=None
        )
        self.HebEMO_model = HebEMO()
        self.emotions = self.HebEMO_model.emotions

    def get_sentiment(self, text):
        sentiment_scores = self.analyzer(text.splitlines())
        sentiment = max(sentiment_scores[0], key=lambda x: x['score'])['label']
        return sentiment

    def get_emotion(self, text):
        emotion_output = self.HebEMO_model.hebemo(text=text)
        emotion_scores = emotion_output[self.emotions]
        emotion = next((e for e in self.emotions if emotion_scores[e][0] == 1), None)
        return emotion

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        # split the transcript into segments of 10 lines each
        segments = ["\n".join(analyzed.transcript.splitlines()[i:i + 10]) for i in
                    range(0, len(analyzed.transcript.splitlines()), 10)]
        sentiments_and_emotions = []

        for segment in tqdm(segments, desc="Textual Sentiment Analyzer on segments"):
            sentiment = self.get_sentiment(segment)
            emotion = self.get_emotion(segment)
            sentiments_and_emotions.append((sentiment, emotion))
        analyzed.__setattr__("sentiments_and_emotions", sentiments_and_emotions)
        return analyzed


if __name__ == '__main__':
    sa = TextualSentimentAnalyzer("sentiment_analyzer")
    analyzed_audio = AnalyzedAudio("analyzed_audio")
    analyzed_output = sa(analyzed_audio)
