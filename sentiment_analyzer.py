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
            "sentiment-analysis",
            model="avichr/heBERT_sentiment_analysis",
            tokenizer="avichr/heBERT_sentiment_analysis",
            return_all_scores=True
        )
        self.HebEMO_model = HebEMO()
        self.emotions = ['anticipation', 'joy', 'trust', 'fear', 'surprise', 'anger', 'sadness', 'disgust']

    def get_sentiment(self, text):
        sentiment_scores = self.analyzer(text)
        sentiment = max(sentiment_scores[0], key=lambda x: x['score'])['label']
        return sentiment

    def get_emotion(self, text):
        emotion_output = self.HebEMO_model.hebemo(text=text)
        emotion_scores = emotion_output[self.emotions]
        emotion = next((e for e in self.emotions if emotion_scores[e][0] == 1), None)
        return emotion

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        segments = analyzed.transcript.splitlines()
        sentiments_and_emotions = []

        for segment in tqdm(segments):
            sentiment = self.get_sentiment(segment)
            emotion = self.get_emotion(segment)
            sentiments_and_emotions.append((sentiment, emotion))
        analyzed.__setattr__("sentiments_and_emotions", sentiments_and_emotions)
        return analyzed


if __name__ == '__main__':
    sa = TextualSentimentAnalyzer("sentiment_analyzer")
    analyzed_audio = AnalyzedAudio("analyzed_audio")
    analyzed_output = sa(analyzed_audio)
