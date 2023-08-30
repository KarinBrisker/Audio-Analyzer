import warnings
from datetime import timedelta
import torch
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import whisper

model = whisper.load_model("large")  # Change this to your desired model
print("Whisper model loaded.")

"""
tone classification
sentiment analysis
toxic words
background noise
"""
def transcribe_audio(path, file_name):
    transcribe = model.transcribe(audio=path, language="he")
    segments = transcribe['segments']

    speech_segments = [segment for segment in segments if segment['no_speech_prob'] < 0.5]
    sentiment = analyze_sentiment(speech_segments)
    classify_audio(segments)

    generate_srt_file(segments, file_name)

    del model.encoder
    del model.decoder
    torch.cuda.empty_cache()
    return f"{file_name}.srt"


def analyze_sentiment(segments):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = []
    for segment in segments:
        text = segment['text']
        blob = TextBlob(text)
        sentiment.append(analyzer.polarity_scores(text))
    return sentiment


def classify_audio(speech_segments):
    # Load the audio data and extract features
    # TODO:
    # https://github.com/openai/whisper/discussions/673
    X = []
    for segment in speech_segments:
        file_path = segment['file_path']  # Add this line to get the file path
        audio, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(audio, sr=sr)
        mel = librosa.feature.melspectrogram(audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(audio, sr=sr)
        features = np.concatenate([mfccs, chroma, mel, contrast])
        X.append(features)
    X = np.array(X)

    # Load the pre-trained classifier
    clf = RandomForestClassifier()
    clf.load('classifier.pkl')

    # Classify the audio segments
    for i, segment in enumerate(speech_segments):
        label = clf.predict(X[i])
        print(f"Background noise: {label}")


def generate_srt_file(segments, file_name):
    with open(f"{file_name}.srt", 'w', encoding='utf-8') as srtFile:
        for segment in segments:
            startTime = str(0) + str(timedelta(seconds=int(segment['start']))) + ',000'
            endTime = str(0) + str(timedelta(seconds=int(segment['end']))) + ',000'
            text = segment['text']
            segmentId = segment['id'] + 1
            segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"
            srtFile.write(segment)


if __name__ == "__main__":
    path_ = "resources/samples/nunu_short.mp3"
    transcribe_audio(path_, "nunu_short")
