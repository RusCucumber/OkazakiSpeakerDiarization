from pathlib import Path

from pydub import AudioSegment

DATA_DIR = Path("data")

def is_wav_file(audio_file_path: Path) -> bool:
    suffix = audio_file_path.suffix
    return suffix == ".wav"

def convert_and_get_wav_path(audio_file_path: Path) -> Path:
    filename = audio_file_path.stem
    wav_path = DATA_DIR / f"processed/{filename}.wav"

    audio = AudioSegment.from_file(audio_file_path)
    audio.export(wav_path, format="wav")

    return wav_path