from analyzed_audio import AnalyzedAudio
from pipeline import Pipe
from simple_diarizer.diarizer import Diarizer


class SpeakerDiarization(Pipe):
    """
    Speaker diarization is the process of partitioning an input audio stream into homogeneous segments according to
    the speaker identity. The output of the speaker diarization process is a list of segments, each segment
    corresponding to a homogeneous region of the audio stream. Each segment is associated with a speaker label.
    --
    https://picovoice.ai/blog/speaker-diarization-in-python/
    https://github.com/facebookresearch/svoice
    --
    https://colab.research.google.com/github/pyannote/pyannote-audio/blob/develop/tutorials/intro.ipynb#scrollTo=lUq1UvoJYnqB
    https://huggingface.co/pyannote/speaker-diarization
    """

    def __init__(self, name):
        super().__init__(name=name)
        # load environment variable
        self.diarization = Diarizer(embed_model='xvec', cluster_method='sc')
        self.num_speakers = 3

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        # Replace "${AUDIO_FILE_PATH}" with the path to your audio file
        segments = self.diarization.diarize(str(analyzed.path))
        for turn in segments:
            print(f"start={turn['start']:.1f}s stop={turn['end']:.1f}s speaker_{turn['label']}")
        analyzed.__setattr__("diarization", self.diarization)
        return analyzed
