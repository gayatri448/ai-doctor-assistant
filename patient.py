# Optional: Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import logging
import os
from io import BytesIO
import speech_recognition as sr
from pydub import AudioSegment
from groq import Groq  # Groq client for STT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ------------------- Step 1: Record Audio -------------------
def record_audio(file_path: str, timeout: int = 20, phrase_time_limit: int = None):
    """
    Records audio from the microphone and saves as MP3.
    
    Args:
        file_path (str): Path to save the recorded audio.
        timeout (int): Max wait time for speech start.
        phrase_time_limit (int): Max duration for recording.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            logging.info(f"Audio saved to {file_path}")
            
    except Exception as e:
        logging.error(f"Recording failed: {e}")


# ------------------- Step 2: Transcribe Audio with Groq -------------------
def transcribe_with_groq(audio_filepath: str, stt_model: str = "whisper-large-v3", GROQ_API_KEY: str = None):
    """
    Transcribes an audio file using Groq STT API.
    
    Args:
        audio_filepath (str): Path to MP3 audio file.
        stt_model (str): Groq STT model to use.
        GROQ_API_KEY (str): API key (optional, read from environment if not provided).
    
    Returns:
        str: Transcribed text.
    """
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment or function argument.")
    
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq client: {e}")
    
    try:
        with open(audio_filepath, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=f,
                language="en"
            )
        return transcription.text
    except Exception as e:
        logging.error(f"STT transcription failed: {e}")
        return "Could not transcribe audio."
        

# ------------------- Example Usage -------------------
if __name__ == "__main__":
    audio_file = "patient_test.mp3"
    # record_audio(audio_file)
    # text = transcribe_with_groq(audio_file)
    # print("Transcribed text:", text)
