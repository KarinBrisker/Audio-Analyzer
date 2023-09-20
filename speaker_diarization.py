import os

from analyzed_audio import AnalyzedAudio
from pipeline import Pipe
from pyannote.audio import Pipeline


class SpeakerDiarization(Pipe):
    """
    Speaker diarization is the process of partitioning an input audio stream into homogeneous segments according to
    the speaker identity. The output of the speaker diarization process is a list of segments, each segment
    corresponding to a homogeneous region of the audio stream. Each segment is associated with a speaker label.
    https://github.com/facebookresearch/svoice
    --
    https://colab.research.google.com/github/pyannote/pyannote-audio/blob/develop/tutorials/intro.ipynb#scrollTo=lUq1UvoJYnqB
    https://huggingface.co/pyannote/speaker-diarization
    """

    def __init__(self, name):
        super().__init__(name=name)
        # load environment variable
        self.sd_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                                    use_auth_token=os.environ.get("HF_TOKEN"))

    def __call__(self, analyzed: AnalyzedAudio) -> AnalyzedAudio:
        diarization = self.sd_pipeline(analyzed.path, num_speakers=4)
        # 5. print the result
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        analyzed.__setattr__("diarization", diarization)
        return analyzed
