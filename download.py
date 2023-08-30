# from datetime import timedelta
# import os
# import whisper
# from pytube import Playlist
# from pytube import YouTube
# import torch


# """
# sampling_rate = 16000
# fine tuning on יניר מרמור
# איך אני יוצר את האודיו ידנית:
# ffmpeg -i input.mp4 -f segment -segment_time 10 -c:a pcm_s16le -ar 44100 -ac 1 output_%03d.wav
# ניקוי קטעים מתים
# חלוקה לדוברים

# [11:28, 23.8.2023] Karin Brisker: אינדיקציה, טון
# [11:28, 23.8.2023] Karin Brisker: רעשי רקע
# [11:29, 23.8.2023] Karin Brisker: סלברייט
# [11:38, 23.8.2023] Karin Brisker: נעה - איפה נמצאים מבחינת פיתוח
# """


# def transcribe_audio(path, file_name):
#     model = whisper.load_model("large")  # Change this to your desired model
#     print("Whisper model loaded.")
#     transcribe = model.transcribe(audio=path)
#     segments = transcribe['segments']

#     for segment in segments:
#         startTime = str(0) + str(timedelta(seconds=int(segment['start']))) + ',000'
#         endTime = str(0) + str(timedelta(seconds=int(segment['end']))) + ',000'
#         text = segment['text']
#         segmentId = segment['id'] + 1
#         segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"

#         srtFilename = f'{file_name}.srt'
#         with open(srtFilename, 'a', encoding='utf-8') as srtFile:
#             srtFile.write(segment)

#     del model.encoder
#     del model.decoder
#     torch.cuda.empty_cache()
#     return srtFilename


# def main():
#     audio_url = "resources/samples/nunu_short"
#     transcribe_audio(f"{audio_url}.mp3", audio_url)

# if __name__ == "__main__":
#     main()
