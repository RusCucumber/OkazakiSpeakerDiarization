from typing import *
from pathlib import Path
import re
from copy import deepcopy

import pandas as pd

PUNCTUATIONS = [".", ",", "!", "?", "-", "%", ":"]

def load_transcript_from_revtxt(txt_path: Path) -> pd.DataFrame:
    df_transcript = pd.read_table(
        txt_path, 
        sep="    ", 
        header=None,
        names=["Speaker", "Timestamp", "Text"],
        engine="python"
    )

    return df_transcript

def replace_tokens_by_regex(
        df_transcript: pd.DataFrame, 
        regex_pattern: str,
        replaced_token: str,
        utterance_col: str ="Text"
) -> pd.DataFrame:
    df_transcript_cleaned = df_transcript.copy(deep=True)
        
    s_utterance = df_transcript_cleaned[utterance_col]
    s_utterance = s_utterance.apply(
        lambda utterance: re.sub(regex_pattern, replaced_token, utterance)
    )

    df_transcript_cleaned.loc[:, utterance_col] = s_utterance

    return df_transcript_cleaned

def modify_tokens_by_func(
        df_transcript: pd.DataFrame, 
        func: Callable,
        utterance_col: str ="Text",
) -> pd.DataFrame:
    df_transcript_cleaned = df_transcript.copy(deep=True)
    
    s_utterance = df_transcript_cleaned[utterance_col]
    s_utterance = s_utterance.apply(func)
    
    df_transcript_cleaned.loc[:, utterance_col] = s_utterance

    return df_transcript_cleaned

def lower_utterance(
        df_transcript: pd.DataFrame, 
        utterance_col: str ="Text"
) -> pd.DataFrame:
    df_transcript_lower = df_transcript.copy(deep=True)

    s_utterance_lower = df_transcript_lower[utterance_col].str.lower()

    df_transcript_lower.loc[:, utterance_col] = s_utterance_lower

    return df_transcript_lower

def number_2_startoken(
        df_transcript: pd.DataFrame,
        utternace_col: str ="Text"
) -> pd.DataFrame:
    number_pattern = r"\d+"
    df_transcript_w_startoken = replace_tokens_by_regex(
        df_transcript,
        number_pattern,
        replaced_token="*",
        utterance_col=utternace_col
    )

    return df_transcript_w_startoken

def remove_punctuation(
        df_transcript: pd.DataFrame,
        utterance_col: str ="Text"
) -> pd.DataFrame:
    def punct_remover(utterance: str) -> str:
        utterance_cleaned = deepcopy(utterance)
        for punct in PUNCTUATIONS:
            utterance_cleaned = utterance_cleaned.replace(punct, "")
        return utterance_cleaned
    
    df_transcript_cleaned = modify_tokens_by_func(
        df_transcript,
        punct_remover,
        utterance_col=utterance_col
    )

    return df_transcript_cleaned

def remove_double_spaces(
        df_transcript: pd.DataFrame, 
        utterance_col: str ="Text"
) -> pd.DataFrame:    
    def double_space_remover(utterance: str) -> str:
        utterance_cleaned = deepcopy(utterance)
        while "  " in utterance_cleaned:
            utterance_cleaned = utterance_cleaned.replace("  ", " ")

        if len(utterance_cleaned) == 0:
            return utterance_cleaned
        
        if utterance_cleaned[0] == " ":
            utterance_cleaned = utterance_cleaned[1:]

        if len(utterance_cleaned) == 0:
            return utterance_cleaned

        if utterance_cleaned[-1] == " ":
            utterance_cleaned = utterance_cleaned[:-1]

        return utterance_cleaned

    df_transcript_cleaned = modify_tokens_by_func(
        df_transcript,
        double_space_remover,
        utterance_col
    )

    return df_transcript_cleaned

def inaudible_2_startoken(
        df_transcript: pd.DataFrame,
        utternace_col: str ="Text"
) -> pd.DataFrame:
    inaudible_tag_pattern = r"\<inaudible\>"
    df_transcript_w_startoken = replace_tokens_by_regex(
        df_transcript,
        inaudible_tag_pattern,
        replaced_token="*",
        utterance_col=utternace_col
    )

    return df_transcript_w_startoken

def inaudible_2_startoken(
        df_transcript: pd.DataFrame,
        utternace_col: str ="Text"
) -> pd.DataFrame:
    inaudible_tag_pattern = r"\<inaudible\>"
    df_transcript_w_startoken = replace_tokens_by_regex(
        df_transcript,
        inaudible_tag_pattern,
        replaced_token="*",
        utterance_col=utternace_col
    )

    return df_transcript_w_startoken

def remove_tags(
        df_transcript: pd.DataFrame,
        utternace_col: str ="Text"
) -> pd.DataFrame:
    tag_pattern = r"\<.*?\>"
    df_transcript_w_startoken = replace_tokens_by_regex(
        df_transcript,
        tag_pattern,
        replaced_token="",
        utterance_col=utternace_col
    )

    return df_transcript_w_startoken

def to_tokens(
        df_transcript: pd.DataFrame,
        utterance_col: str ="Text"
) -> List[str]:
    tokens = []
    for idx in df_transcript.index:
        utterance = df_transcript.at[idx, utterance_col]
        tokens += utterance.split(" ")

    return tokens

def to_str(
        df_transcript: pd.DataFrame,
        utterance_col: str ="Text"
) -> str:
    return " ".join(df_transcript[utterance_col])

def convert_turnwise(
        df: pd.DataFrame, 
        speaker_col: str ="speaker", 
        end_time_col: str ="end_time", 
        transcript_col: str ="transcript"
) -> pd.DataFrame:
    data = []
    columns = df.columns.values

    prev_speaker = df.at[0, speaker_col]
    utterances = []
    for _, row in df.iterrows():
        if prev_speaker == getattr(row, speaker_col):
            utterances.append(row)
        else:
            new_row = generate_new_row(utterances, columns, end_time_col, transcript_col)
            data.append(new_row)
            utterances = [row]
        prev_speaker = getattr(row, speaker_col)

    new_row = generate_new_row(utterances, columns, end_time_col, transcript_col)
    data.append(new_row)

    return pd.DataFrame(data, columns=columns)


def generate_new_row(
        utterances: list, 
        columns: list, 
        end_time_col: str ="end_time", 
        transcript_col: str ="transcript"
) -> list:
    row = []
    for col in columns:
        if col == end_time_col:
            val = getattr(utterances[-1], col)
        elif col == transcript_col:
            val = " ".join([getattr(u, col) for u in utterances if str(getattr(u, col)) != "nan"])
        else:
            val = getattr(utterances[0], col)
            
        row.append(val)

    return row