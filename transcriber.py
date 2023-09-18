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

    def transcribe_audio(self, path, file_name):
        transcribe = self.model.transcribe(audio=path, language="he", fp16=False)
        segments = transcribe['segments']
        self.generate_srt_file(segments, file_name)
        return f"{file_name}.srt"

    @staticmethod
    def generate_srt_file(segments, file_name):
        with open(f"{file_name}.srt", 'w', encoding='utf-8') as srtFile:
            for segment in segments:
                start_time = str(0) + str(timedelta(seconds=int(segment['start']))) + ',000'
                end_time = str(0) + str(timedelta(seconds=int(segment['end']))) + ',000'
                text = segment['text']
                segment_id = segment['id'] + 1
                segment = f"{segment_id}\n{start_time} --> {end_time}\n{text[1:] if text[0] == ' ' else text}\n\n"
                srtFile.write(segment)

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        transcribe = self.model.transcribe(audio=analyzed.audio, language="he", fp16=False)
        segments = transcribe['segments']
        analyzed.transcript = segments
        return analyzed

