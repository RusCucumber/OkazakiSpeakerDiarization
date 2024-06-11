from typing import Tuple
from pathlib import Path
from argparse import ArgumentParser
import sys, json

sys.path.append(
    str(Path.cwd())
)

import torch
import pyannote
from pyannote.audio import Pipeline

from python.modules.audio_utils import is_wav_file, convert_and_get_wav_path

RESULT_DIR = Path("result")

PYANNOTE_ACCESS_TOKEN_PATH = Path("environment/python/pyannote_access_token.json")

def read_arguments() -> Tuple[Path, Path, str]:
    arg_parser = ArgumentParser(description="This is Pyannote Speaker Diarization Program")

    arg_parser.add_argument("--data_path", type=str)
    arg_parser.add_argument("--save_dir", type=str, default=str(RESULT_DIR))
    arg_parser.add_argument("--device", type=str, default="cpu")

    args = arg_parser.parse_args()

    data_path = Path(args.data_path)
    save_dir = Path(args.save_dir)
    device = args.device

    return data_path, save_dir, device

def diarize(data_path: Path, device: str) -> pyannote.core.annotation.Annotation:
    pipelien = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=auth_token["pyannoteAccessToken"]
    )

    pipelien.to(torch.device(device))

    diarization = pipelien(data_path)

    return diarization

if __name__ == "__main__":
    with open(PYANNOTE_ACCESS_TOKEN_PATH, "r") as f:
        auth_token = json.load(f)

    data_path, save_dir, device = read_arguments()

    if not is_wav_file(data_path):
        data_path = convert_and_get_wav_path(data_path)

    diarization = diarize(data_path, device)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")