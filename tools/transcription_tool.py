import os
import shutil
import subprocess
from fastapi import UploadFile
from transformers import pipeline
import torch

# --- Configuration ---
MODEL_NAME = "openai/whisper-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMP_AUDIO_DIR = "temp_audio"

# --- Singleton pattern for loading the model ---
TRANSCRIBER = None


def check_ffmpeg() -> bool:
    """Verify ffmpeg is installed and accessible in the system's PATH."""
    try:
        # Use a command that is quiet and exits quickly
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def load_transcriber():
    """Initializes and returns the ASR pipeline."""
    global TRANSCRIBER
    if TRANSCRIBER is None:
        if not check_ffmpeg():
            # Provide a clear error message if ffmpeg is not found.
            raise RuntimeError("FFmpeg not found. Please install ffmpeg and ensure it is in your system's PATH.")

        print(f"INFO: Loading transcription model '{MODEL_NAME}' on device '{DEVICE}'...")
        TRANSCRIBER = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            device=DEVICE
        )
        print("INFO: Transcription model loaded successfully.")
    return TRANSCRIBER


def transcribe_audio(file: UploadFile) -> str:
    """
    Transcribes an audio file to text using the Whisper model.

    Args:
        file: The uploaded audio file from the FastAPI endpoint.

    Returns:
        The transcribed text as a string, or an error message.
    """
    try:
        transcriber = load_transcriber()
    except Exception as e:
        print(f"ERROR loading transcriber: {e}")
        return f"Error: {e}"

    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    temp_file_path = os.path.join(TEMP_AUDIO_DIR, file.filename or "temp_audio_file")

    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe the audio file
        result = transcriber(
            temp_file_path,
            generate_kwargs={"task": "transcribe"}  # auto-detects language
        )

        transcribed_text = result.get("text", "").strip()
        if not transcribed_text:
            return "Error: Could not extract any text from the audio. The file might be silent or unsupported."

        return transcribed_text

    except Exception as e:
        print(f"ERROR during transcription: {e}")
        return f"Error during transcription: {e}"

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# Example for direct testing
if __name__ == '__main__':
    print("This module is intended to be imported and used in the FastAPI app.")
    print("Checking for ffmpeg...")
    if check_ffmpeg():
        print("✅ FFmpeg found.")
    else:
        print("❌ FFmpeg not found. Please install it and add to your system's PATH.")
