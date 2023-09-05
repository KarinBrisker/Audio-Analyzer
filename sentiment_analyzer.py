from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text):
        blob = TextBlob(text)
        sentiment = self.analyzer.polarity_scores(text)
        return sentiment

    def analyze_segments(self, segments):
        sentiments = []
        for segment in segments:
            text = segment['text']
            sentiments.append(self.analyze(text))
        return sentiments
