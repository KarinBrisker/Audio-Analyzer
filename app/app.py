import gradio as gr
import whisper
from gradio.components import Audio, Textbox

# gradio app.py my_demo


def transcribe_audio(audio_file):
    status_text = "Audio file uploaded successfully!"
    yield "", status_text

    status_text = "Loading model..."
    yield "", status_text
    model = whisper.load_model("base")
    status_text = "Model loaded successfully!"
    yield "", status_text

    status_text = "Transcribing audio..."
    yield "", status_text
    result = model.transcribe(audio_file)
    status_text = "Transcription finished!"
    yield result["text"], status_text


def main():
    audio_input = Audio(source="upload", type="filepath")
    output_text = Textbox(label="Transcribed Text")
    status_text = Textbox(label="Status")

    iface = gr.Interface(fn=transcribe_audio, inputs=audio_input,
                         outputs=[output_text, status_text], title="Your Transcription Analyzer",
                         description="Upload an audio file and hit the 'Submit'\
                             button to transcribe the audio into text.",
                         theme=gr.themes.Soft())

    iface.queue()
    iface.launch()


if __name__ == '__main__':
    main()
