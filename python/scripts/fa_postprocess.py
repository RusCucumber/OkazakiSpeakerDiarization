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

    arg_parser.add_argument(
        "-o", 
        "--original_transcript_path", 
        type=str
    )
    arg_parser.add_argument(
        "-f", 
        "--forced_alignment_path", 
        type=str
    )
    arg_parser.add_argument(
        "--save_dir", 
        type=str, 
        default=str(DATA_DIR / "processed")
    )

    args = arg_parser.parse_args()

    transcript_path = Path(args.original_transcript_path)
    fa_path = Path(args.forced_alignment_path)
    save_dir = Path(args.save_dir)

    return transcript_path, fa_path, save_dir

def modify_inaudible_tag(
        df_transcript: pd.DataFrame,
        utternace_col: str ="Text"
) -> pd.DataFrame:
    inaudible_tag_pattern = r"\<inaudible\>"
    df_transcript_w_mod_inaudible = replace_tokens_by_regex(
        df_transcript,
        inaudible_tag_pattern,
        replaced_token="[inaudible]",
        utterance_col=utternace_col
    )

    return df_transcript_w_mod_inaudible

def preprocess(df_transcript: pd.DataFrame) -> pd.DataFrame:
    df_transcript = modify_inaudible_tag(df_transcript)

    df_transcript = remove_punctuation(df_transcript)
    df_transcript = remove_tags(df_transcript)
    df_transcript = remove_double_spaces(df_transcript)

    df_transcript = lower_utterance(df_transcript)

    mask_blank = (df_transcript["Text"] == "")
    df_transcript = df_transcript[~mask_blank]

    return df_transcript

def fill_startoken(
        df_transcript: pd.DataFrame, 
        df_fa: pd.DataFrame
) -> pd.DataFrame:
    transcript = to_str(df_transcript)
    words = transcript.split(" ")

    if len(words) != len(df_fa):
        raise RuntimeError(f"The length of transcripts is different.")

    df_fa_star_filled = df_fa.copy(deep=True)
    df_fa_star_filled.loc[:, "word"] = words

    return df_fa_star_filled

if __name__ == "__main__":
    transcript_path, fa_path, save_dir = read_arguments()
    df_transcript = load_transcript_from_revtxt(transcript_path)
    df_transcript = preprocess(df_transcript)
    df_fa = pd.read_csv(fa_path)

    df_fa = fill_startoken(df_transcript, df_fa)

    save_path = save_dir / f"{transcript_path.stem}_fa.csv"
    df_fa.to_csv(save_path, index=False)