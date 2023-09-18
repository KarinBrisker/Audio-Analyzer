from pydub import AudioSegment


class AudioEnhancer:
    def __call__(self, input_audio_path: str, output_audio_path: str, volume: int = 10) -> None:
        sound = AudioSegment.from_file(input_audio_path)
        louder_sound = sound + volume
        louder_sound.export(output_audio_path, format="wav")


if __name__ == "__main__":
    audio_enhancer = AudioEnhancer()
    audio_enhancer("audio_output\REC00 3_clean.wav", "audio_output\REC00 3_louder.wav")
