from dotenv import load_dotenv
load_dotenv()

import os
from gtts import gTTS

def text_to_speech(input_text, output_filepath):
    """
    Simple TTS using gTTS.
    Saves speech to an MP3 file.
    """
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts.save(output_filepath)
    return output_filepath


# Example usage:
if __name__ == "__main__":
    input_text = "Hi, this is AI Doctor Assistant testing TTS!"
    output_file = "doctor_response.mp3"
    text_to_speech(input_text, output_file)
    print(f"Saved TTS audio to {output_file}")
