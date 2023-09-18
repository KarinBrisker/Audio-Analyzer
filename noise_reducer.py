import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np

from pipeline import Pipe


class NoiseReducer(Pipe):
    def __call__(self, input_audio_path: str, duration: float = None) -> (np.ndarray, int, np.ndarray):
        audio_, sr_ = librosa.load(input_audio_path, sr=None, duration=duration)
        int_sr_ = int(sr_)
        clean_audio = nr.reduce_noise(y=audio_, sr=int_sr_, stationary=False)
        return clean_audio, int_sr_, audio_


if __name__ == "__main__":
    noise_reducer = NoiseReducer()
    cleaner, sr, audio = noise_reducer("resources\REC00 3.wav")
    sf.write("audio_output\REC00 3_original.wav", audio, sr)
    sf.write("audio_output\REC00 3_clean.wav", cleaner, sr)
