import streamlit as st
import soundfile as sf

from time_utils import TimeUtils
from app_utils import get_problematic_sections, transcribe_audio, html_formatter
from styling import custom_css, tag_colors

# Inject custom CSS styles into Streamlit
st.markdown(custom_css, unsafe_allow_html=True)

current_section_index = st.empty()

# Initialize placeholders for the answer buttons
yes_button = st.empty()
no_button = st.empty()


# Function to display a question and answer buttons
def display_section(audio, index, sections):
    section = sections[index]
    start = section.get("start", 0)
    end = section.get("end", 0)
    description = section.get("description", "")
    st.write(f"Question {st.session_state.current_question_index + 1}/{len(sections)}")
    # Display the question in bigger font than the buttons
    st.header("Is the following record problematic?")
    tags = description.split(", ")
    st.markdown("<h5 style='text-align: left; color: brown;'>Detected Signals:</h5>",
                unsafe_allow_html=True)
    # Create len(tags) columns for the tags
    cols = st.columns(len(tags))
    for i, tag in enumerate(tags):
        cols[i].markdown(
            f'<span style="color: white; background-color: {tag_colors[tag]}; padding: 10px; border-radius: '
            f'5px; display: inline-block; margin-right: 5px;">{tag}</span>',
            unsafe_allow_html=True)

    # Display the audio section

    start_time = int(TimeUtils.time_string_to_seconds(start))
    end_time = int(TimeUtils.time_string_to_seconds(end))

    audio.seek(start_time)
    sub_audio = audio.read(end_time - start_time)

    sample_rate = audio.samplerate  # get sample rate from SoundFile object
    st.audio(sub_audio, sample_rate=sample_rate)


    # Generate unique keys for the "Yesx" and "No" buttons
    yes_key = f"yes-{index}-{sections[index]}"
    no_key = f"no-{index}-{sections[index]}"

    # Create two columns for the "Yes" and "No" buttons
    col1, col2 = st.columns(2)

    # Create "Yes" button with custom style
    if col1.button("Yes", key=yes_key, on_click=handle_yes_click, type="primary"):
        move_to_next_question(sections)

    # Create "No" button with custom style
    if col2.button("No", key=no_key, on_click=handle_no_click, type="primary"):
        move_to_next_question(sections)

    # Return the button states
    return st.session_state[yes_key], st.session_state[no_key]


def handle_yes_click():
    # Handle "Yes" button click event here
    # For example, you can record the user's choice or perform other actions
    pass


def handle_no_click():
    # Handle "No" button click event here
    # For example, you can record the user's choice or perform other actions
    pass


def move_to_next_question(sections):
    # Move to the next question by incrementing the current question index
    st.session_state.current_question_index += 1

    # Reset the question index to 0 if we have reached the end of the questions list
    if st.session_state.current_question_index >= len(sections):
        st.session_state.current_question_index = 0


def main():
    # title in center
    st.markdown("<h1 style='text-align: center; color: brown;'>Your Child Kinder Guard 💂‍♀️🧸</h1>",
                unsafe_allow_html=True)
    # add a subheader
    st.markdown("<h5 style='text-align: center; color: black;'>In collaboration with המטה למאבק למען הילדים</h2>",
                unsafe_allow_html=True)

    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("app/resources/icon.png", width=170)

    st.write("Upload an audio file and hit the 'Submit'\
                             button to transcribe the audio into text.")

    audio_file = st.file_uploader("Upload Audio 🎧", type=['mp3', 'wav'])

    if audio_file is not None:
        language_codes = st.multiselect("Select Language(s)", ["en", "he", "ar", "es", "fr", "ru"])
        child_name = st.text_input("Child Name")

        if language_codes:
            st.write("You chose: " + ", ".join(language_codes) + " 🗣️", default="en")

        transcript = transcribe_audio()
        formatted_text = html_formatter(transcript, child_name)
        with st.expander("Transcript", expanded=False):
            st.markdown(f'<div style="direction: rtl; text-align: right;">{formatted_text}</div>',
                        unsafe_allow_html=True)
        with st.expander("Problematic Sections 🤔", expanded=True):
            sections = get_problematic_sections()
            st.write(len(sections))
            audio_path = "resources/samples/REC00_3.wav"
            audio = sf.SoundFile(audio_path)
            # Start with the first question

            # Initialize the current question index to 0
            if "current_question_index" not in st.session_state:
                st.session_state.current_question_index = 0
            display_section(audio, st.session_state.current_question_index, sections)


if __name__ == '__main__':
    main()
