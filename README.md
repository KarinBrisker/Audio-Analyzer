## TODO:
run pipreqs to generate requirements.txt
```bash
pipreqs . --force
```

### list of tasks:
#### audio:
- [X] background noise classification
- [ ] audio enhancement
     1. **Librosa**: It's a Python library for audio and music analysis. It provides the building blocks necessary to create music information retrieval systems.

     2. **Noisereduce**: This library performs noise reduction using spectral gating in python. It can be very effective for certain types of consistent noise such as white noise, ambient noise, or background chatter.

     3. **Scipy**: Scipy has a module signal which has functions like spectrogram for visualizing time-frequency representation and butter, lfilter for designing and applying a digital filter.

     4. **Pydub**: Pydub is a simple and easy-to-use Python library for audio manipulation. It can be used to adjust the volume of an audio file, which can help in normalizing the sound.

- [ ] speaker diarization  # [Link1](https://github.com/facebookresearch/svoice), [Link2](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-stt-diarization?tabs=windows&pivots=programming-language-python)
- [ ] cleaning dead segments  # risky?
- [ ] tone classification

#### text:
- [X] text extraction
- [X] sentiment analysis
- [ ] toxic words


### GENERAL NOTES:
- sampling rate: 16000

### data:
- [ ] [FSD50K](https://annotator.freesound.org/fsd/release/FSD50K/)
- [ ] [AudioSet](https://research.google.com/audioset/)

### models:
- [ ] [CLAP](https://arxiv.org/pdf/2206.04769)

