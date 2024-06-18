from typing import *
from math import isclose

import pandas as pd
from textgrids import TextGrid, Tier, Interval

def generate_intervals(
        value: Any,
        prev_end_time: float,
        start_time: float,
        end_time: float        
) -> List[Interval]:
    interval = Interval(text=str(value), xmin=start_time, xmax=end_time)
    if prev_end_time == start_time:
        return [interval]
    
    blank_interval = Interval(text="", xmin=prev_end_time, xmax=start_time)
    return [blank_interval, interval]

def df_2_interval_tier(
        df: pd.DataFrame,
        target_col: str,
        start_time_col: str,
        end_time_col: str,
        tier_end_time: Optional[float] =None
) -> Tier:
    tier = []
    prev_end_time = 0.0
    for idx in df.index:
        value = df.at[idx, target_col]
        start_time = df.at[idx, start_time_col]
        end_time = df.at[idx, end_time_col]

        tier += generate_intervals(
            value, 
            prev_end_time, 
            start_time,
            end_time
        )

        prev_end_time = end_time

    if tier_end_time is None:
        return Tier(tier, xmin=0.0, xmax=end_time)

    if isclose(end_time, tier_end_time, rel_tol=0.001):
        return Tier(tier, xmin=0.0, xmax=tier_end_time)

    tier.append(
        Interval("", xmin=end_time, xmax=tier_end_time)
    )

    return Tier(tier, xmin=0.0, xmax=tier_end_time)