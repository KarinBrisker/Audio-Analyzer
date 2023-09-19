# Audio Analyzer
### Description:
Audio Analyzer is a pipeline for audio analysis that aims to provide a comprehensive solution for audio processing. The project is designed to be modular and scalable, allowing users to easily add new features and functionalities.

### Solution Diagram:

![pipeline_diagram](https://github.com/KarinBrisker/audio_analyzer/assets/19929107/d8813349-42e0-4a1a-8f16-a3ee9cdfae60)


### list of tasks:
- verify each step
- not assuming language
- using CLAP model
- check the model on very long audio input = scale [~8 hours]

#### audio:
- [X] background noise classification
- [X] audio enhancement
- [ ] speaker diarization [Link1](https://github.com/facebookresearch/svoice), [Link2](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-stt-diarization?tabs=windows&pivots=programming-language-python)
- [ ] tone classification


#### text:
- [X] text extraction
- [X] sentiment analysis
- [ ] toxic words detection


---
#### future tasks:
- [ ] text summarization
- [ ] cleaning dead segments
- [ ] audio segmentation - part of day

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
2. run:
    ```bash
    pip install -r requirements.txt
    ```
