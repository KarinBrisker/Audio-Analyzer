# Audio Analyzer
### Description:
This project is a pipeline for audio analysis.

### Solution Diagram:

![pipeline_diagram](https://github.com/KarinBrisker/audio_analyzer/assets/19929107/d8813349-42e0-4a1a-8f16-a3ee9cdfae60)


### list of tasks:
#### audio:
- [X] background noise classification
- [X] audio enhancement
- [ ] speaker diarization [Link1](https://github.com/facebookresearch/svoice), [Link2](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-stt-diarization?tabs=windows&pivots=programming-language-python)
- [ ] cleaning dead segments  -  risky?
- [ ] tone classification

#### text:
- [X] text extraction
- [X] sentiment analysis
- [ ] toxic words
- [ ] text summarization

### data:
- [ ] [FSD50K](https://annotator.freesound.org/fsd/release/FSD50K/)
- [ ] [AudioSet](https://research.google.com/audioset/)

### models:
- [ ] [CLAP](https://arxiv.org/pdf/2206.04769)

### other:
1. run pipreqs to generate requirements.txt
```bash
pipreqs . --force
```
1. run:
```bash
pip install -r requirements.txt
```