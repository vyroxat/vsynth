# VSynth

VSynth is a desktop application that transforms human speech into a retro, robotic formant synthesizer voice. It allows you to either record audio directly from your microphone or load an existing audio file, and converts the spoken words into an artificial voice in real-time.

## How It Works

1. Transcription: The application uses Faster-Whisper, an optimized, offline speech recognition engine. It includes Voice Activity Detection to filter out background music or noise, ensuring only clear vocals are transcribed.
2. Phoneme Conversion: The transcribed text is converted into ARPABET phonetic representations.
3. Synthesis: The phonemes are passed into the Klattsch formant synthesizer, which generates the retro robotic audio.

## Features

- Microphone Recording: Record your voice directly within the app.
- Audio File Support: Load common audio formats (MP3, WAV, M4A) for transcription and synthesis.
- Voice Parameters: Adjust the generated voice using built-in sliders for Pitch, Speed, Formant Scale, Vibrato, Aspiration, Spectral Tilt, and Vocal Effort.
- Presets: Save and load custom configurations for the synthesizer sliders.
- Standalone Mode: Fully functional offline. No cloud APIs or external servers are required.
- File Export: Save the final synthesized voice output to a custom WAV file on your computer.

## Requirements

If you are running the source code directly instead of the standalone executable, you need to install the following dependencies:

- Python 3.10+
- Node.js (required for the Klattsch engine via npx)
- FFmpeg (must be installed and added to your system PATH for audio processing)

Dependencies can be installed via:
pip install -r requirements.txt

## Running the Application

To start the application from source, run:
python main.py
