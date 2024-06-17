from typing import *
from pathlib import Path
from argparse import ArgumentParser
import sys

sys.path.append(
    str(Path.cwd())
)

import pandas as pd

from python.modules.transcript_utils import *

DATA_DIR = Path("data")

def read_arguments() -> Tuple[Path, Path]:
    arg_parser = ArgumentParser(
        description="This program generate txt file for wav2vec FA."
    )

    arg_parser.add_argument("--data_path", type=str)
    arg_parser.add_argument(
        "--save_dir", 
        type=str, 
        default=str(DATA_DIR / "processed")
    )

    args = arg_parser.parse_args()

    data_path = Path(args.data_path)
    save_dir = Path(args.save_dir)

    return data_path, save_dir

def preprocess(df_transcript: pd.DataFrame) -> pd.DataFrame:
    df_transcript = number_2_startoken(df_transcript)
    df_transcript = inaudible_2_startoken(df_transcript)

    df_transcript = remove_punctuation(df_transcript)
    df_transcript = remove_tags(df_transcript)
    df_transcript = remove_double_spaces(df_transcript)

    df_transcript = lower_utterance(df_transcript)

    return df_transcript

def write_fa_txt(transcript: str, save_path: Path) -> None:
    with open(save_path, "w") as f:
        f.write(transcript)

if __name__ == "__main__":
    data_path, save_dir = read_arguments()
    df_transcript = load_transcript_from_revtxt(data_path)

    df_transcript = preprocess(df_transcript)
    transcript = to_str(df_transcript)

    save_path = save_dir / f"{data_path.stem}.txt"

    write_fa_txt(transcript, save_path)