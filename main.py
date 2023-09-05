import warnings
from datetime import timedelta
import torch
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from classify import calculate_average_logprobs

import whisper
from whisper.tokenizer import Tokenizer, get_tokenizer

model = whisper.load_model("large")  # Change this to your desired model
tokenizer = get_tokenizer(multilingual=True, language="he")
print("Whisper model loaded.")


def transcribe_audio(path, file_name):
    transcribe = model.transcribe(audio=path, language="he", fp16=False)
    segments = transcribe['segments']
    classify_audio(segments)

    speech_segments = [segment for segment in segments if segment['no_speech_prob'] < 0.5]
    sentiment = analyze_sentiment(speech_segments)

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
    class_names = open("resources/class_names.txt").read().splitlines()
    for segment in speech_segments:
        # Convert the audio segment into a log-Mel spectrogram
        spectrogram = segment['mel']

        # Pass the spectrogram into the encoder of the Whisper model
        audio_features = model.encoder(spectrogram)

        average_logprobs = calculate_average_logprobs(model=model, audio_features=audio_features,
                                                      class_names=class_names, tokenizer=tokenizer)
        sorted_indices = sorted(range(len(class_names)), key=lambda i: average_logprobs[i], reverse=True)

        predicted_class = class_names[average_logprobs.argmax()]
        print(f"Background noise: {predicted_class}")


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
