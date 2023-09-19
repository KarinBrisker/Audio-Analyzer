import os.path
from datetime import timedelta

import whisper
from whisper.tokenizer import get_tokenizer

from analyzed_audio import AnalyzedAudio
from pipeline import Pipe


class WhisperTranscriber(Pipe):
    def __init__(self, name, language="he"):
        super().__init__(name=name)
        whisper_model = "large"
        self.model = whisper.load_model(whisper_model)
        self.tokenizer = get_tokenizer(multilingual=True, language=language)
        print("Whisper model loaded.")

    def transcribe_audio(self, analyzed: AnalyzedAudio):
        transcript = self.model.transcribe(audio=analyzed.audio, language="he", fp16=False)
        segments = transcript['segments']
        analyzed.__setattr__("transcript", transcript.text)
        analyzed.__setattr__("segments", segments)
        transcript_path = os.path.join(analyzed.base_path, analyzed.filename + "_transcript.srt")
        self.generate_srt_file(segments, transcript_path)
        return analyzed

    @staticmethod
    def generate_srt_file(segments, file_path):
        with open(file_path, 'w', encoding='utf-8') as srtFile:
            for segment in segments:
                start_time = str(0) + str(timedelta(seconds=int(segment['start']))) + ',000'
                end_time = str(0) + str(timedelta(seconds=int(segment['end']))) + ',000'
                text = segment['text']
                segment_id = segment['id'] + 1
                segment = f"{segment_id}\n{start_time} --> {end_time}\n{text[1:] if text[0] == ' ' else text}\n\n"
                srtFile.write(segment)

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        analyzed = self.transcribe_audio(analyzed)

        # save srt file

        return analyzed

