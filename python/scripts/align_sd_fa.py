from typing import *
from pathlib import Path
from argparse import ArgumentParser
import sys

sys.path.append(
    str(Path.cwd())
)

import pandas as pd
from pydub import AudioSegment

from python.modules.transcript_utils import *
from python.modules.textgrid_utils import *

DATA_DIR = Path("data")
RESULT_DIR = Path("results/tables")

def read_arguments() -> Tuple[Path, Path, Path, Path, str]:
    arg_parser = ArgumentParser(
        description="This program aligns a speaker diarization result with forced alignment timestamps"
    )

    arg_parser.add_argument("-s", "--sd_path", type=str)
    arg_parser.add_argument("-fs", "--fa_sent_path", type=str)
    arg_parser.add_argument("-fw", "--fa_word_path", type=str)
    arg_parser.add_argument("-a", "--audio_path", type=str)
    arg_parser.add_argument("--save_dir", type=str, default=str(RESULT_DIR))

    args = arg_parser.parse_args()

    sd_path = Path(args.sd_path)
    fa_sent_path = Path(args.fa_sent_path)
    fa_word_path = Path(args.fa_word_path)
    audio_path = Path(args.audio_path)
    save_dir = Path(args.save_dir)

    return sd_path, fa_sent_path, fa_word_path, audio_path, save_dir

def transform_speakerwise(df_sd: pd.DataFrame) -> pd.DataFrame:
    df_sd_speakerwise = df_sd.copy(deep=True) 
    df_sd_speakerwise.loc[:, "transcript"] = ""

    df_sd_speakerwise = convert_turnwise(
        df_sd_speakerwise, 
        speaker_col="Speaker", 
        end_time_col="Stop", 
        transcript_col="transcript"
    )

    return df_sd_speakerwise

def get_audio_duration(audio_path: Path) -> float:
    audio = AudioSegment.from_file(audio_path)
    return audio.duration_seconds

def align_sd_fa(
        df_sd: pd.DataFrame, 
        df_fa_s: pd.DataFrame,
        df_fa_w: pd.DataFrame,
        tier_end_time: float
) -> TextGrid:
    grid = TextGrid()
    tier_text = df_2_interval_tier(
        df_fa_s, target_col="text",
        start_time_col="start_time",
        end_time_col="end_time",
        tier_end_time=tier_end_time
    )
    grid["text"] = tier_text

    tier_word = df_2_interval_tier(
        df_fa_w, target_col="word",
        start_time_col="start_time",
        end_time_col="end_time",
        tier_end_time=tier_end_time
    )
    grid["word"] = tier_word

    for speaker in sorted(set(df_sd["Speaker"])):
        mask_spekaer = (df_sd["Speaker"] == speaker)
        tier_sd = df_2_interval_tier(
            df_sd[mask_spekaer], target_col="Speaker",
            start_time_col="Start",
            end_time_col="Stop",
            tier_end_time=tier_end_time
        )
        grid[speaker] = tier_sd

    grid.xmax = tier_sd.xmax
    return grid


if __name__ == "__main__":
    sd_path, fa_sent_path, fa_word_path, audio_path, save_dir = read_arguments()

    df_sd = pd.read_csv(sd_path)
    df_sd = transform_speakerwise(df_sd)

    df_fa_s = pd.read_csv(fa_sent_path)
    df_fa_w = pd.read_csv(fa_word_path)

    tier_end_time = get_audio_duration(audio_path)

    grid_sd_transcript = align_sd_fa(df_sd, df_fa_s, df_fa_w, tier_end_time)

    save_path = save_dir / f"{sd_path.stem}.TextGrid"
    grid_sd_transcript.write(str(save_path))