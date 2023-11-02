import json
from pydub import AudioSegment

# Open the file and read its contents line by line
with open("./resources/toxic.json", "r", encoding="utf-8") as file:
    toxic_words = json.load(file)


def get_problematic_sections():
    with open("resources/ranker_output.json", "r") as f:
        # load json
        sections = json.loads(f.read())
    return sections


def extract_audio_subsection(audio_file, start_time, end_time):
    audio = AudioSegment.from_file(audio_file)
    start_ms = int(start_time * 1000)  # Convert to milliseconds
    end_ms = int(end_time * 1000)  # Convert to milliseconds
    extracted_audio = audio[start_ms:end_ms]
    return extracted_audio


def html_formatter(transcript, child_name=None):
    formatted_text = ""
    lines = transcript.split("\n")

    for line in lines:
        words = line.split()
        formatted_line = ""
        i = 0

        while i < len(words):
            found_problem = False
            # keep only the word itself, without punctuation
            clean_word = words[i].strip(".,!?")
            # Check for child's name
            if child_name is not None and clean_word == child_name:
                formatted_line += '<span style="font-size: 1.7em; font-weight: bold; color: blue;">' + child_name + '</span> '
                i += 1
                continue
            # Check for problematic phrases
            for phrase in toxic_words:
                phrase_words = phrase.split()
                # remove punctuation from phrase
                phrase_to_check = [word.strip(".,!?") for word in words[i:i + len(phrase_words)]]

                if phrase_to_check == phrase_words:
                    formatted_line += '<span style="font-size: 1.7em; font-weight: bold; color: red;">' + ' '.join(
                        phrase_words) + '</span> '
                    i += len(phrase_words)
                    found_problem = True
                    break

            if not found_problem:
                formatted_line += words[i] + " "
                i += 1

        formatted_text += formatted_line + "<br>"

    return formatted_text


def transcribe_audio():
    with open("resources/transcript+bad_words.txt", "r") as f:
        transcript = f.read()

    return transcript
