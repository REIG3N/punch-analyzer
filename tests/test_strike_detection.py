import math

import pandas as pd
import pytest

from punch_analyzer.strike_detection import (
    LEFT_WRIST_ID,
    RIGHT_WRIST_ID,
    _fill_short_gaps,
    compute_wrist_speed,
    detect_strikes,
    detect_strikes_for_hand,
)

FPS = 30.0
FRAME_MS = 1000.0 / FPS


def _wrist_df(
    landmark_id: int, x_values: list[float], y_values: list[float] | None = None
) -> pd.DataFrame:
    if y_values is None:
        y_values = [0.5] * len(x_values)
    rows = []
    for frame_idx, (x, y) in enumerate(zip(x_values, y_values)):
        rows.append(
            {
                "frame_idx": frame_idx,
                "timestamp_ms": frame_idx * FRAME_MS,
                "landmark_id": landmark_id,
                "x": x,
                "y": y,
                "z": 0.0,
                "visibility": math.nan if math.isnan(x) else 0.9,
            }
        )
    return pd.DataFrame(rows)


def test_fill_short_gaps_interpolates_runs_within_limit():
    series = pd.Series([0.0, math.nan, math.nan, 0.6, 0.8])
    filled = _fill_short_gaps(series, max_gap_frames=2)
    assert filled.iloc[1] == pytest.approx(0.2)
    assert filled.iloc[2] == pytest.approx(0.4)


def test_fill_short_gaps_leaves_long_runs_as_nan():
    series = pd.Series([0.0] + [math.nan] * 8 + [0.9])
    filled = _fill_short_gaps(series, max_gap_frames=5)
    assert filled.iloc[1:9].isna().all()


def test_compute_wrist_speed_no_explosion_across_long_gap():
    # 5 frames stables, 8 frames de tracking perdu (> DEFAULT_MAX_GAP_FRAMES),
    # puis reprise sur une position très différente : ne doit pas produire un
    # pic de vitesse artificiel au moment de la reprise.
    x_values = [0.2] * 5 + [math.nan] * 8 + [0.9] * 5
    df = _wrist_df(LEFT_WRIST_ID, x_values)

    speed = compute_wrist_speed(df, LEFT_WRIST_ID, max_gap_frames=5)

    gap_boundary_speeds = speed["speed"].iloc[5:14]
    assert gap_boundary_speeds.isna().all()
    # Aucune valeur de vitesse ne doit être un nombre gigantesque (explosion).
    assert speed["speed"].dropna().le(50).all()


def test_compute_wrist_speed_interpolates_short_gap_smoothly():
    x_values = [0.2, 0.2, math.nan, math.nan, 0.26, 0.28]
    df = _wrist_df(LEFT_WRIST_ID, x_values)

    speed = compute_wrist_speed(df, LEFT_WRIST_ID, max_gap_frames=3)

    assert speed["speed"].notna().sum() >= 3
    assert speed["speed"].dropna().le(10).all()


def test_detect_strikes_for_hand_finds_single_peak():
    x_values = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.05,
        0.15,
        0.40,
        0.42,
        0.43,
        0.43,
        0.43,
        0.43,
    ]
    df = _wrist_df(LEFT_WRIST_ID, x_values)
    wrist_speed = compute_wrist_speed(df, LEFT_WRIST_ID)

    strikes = detect_strikes_for_hand(
        wrist_speed, hand="left", min_interval_ms=100.0, min_prominence=1.0
    )

    assert len(strikes) == 1
    assert strikes[0].frame_idx == 8
    assert strikes[0].hand == "left"


def test_detect_strikes_for_hand_deduplicates_within_min_interval():
    # Deux mini-pics très rapprochés dans le temps : avec un min_interval_ms
    # large, ils doivent être comptés comme un seul coup (le plus fort retenu).
    x_values = [0.0, 0.3, 0.0, 0.3, 0.0] + [0.0] * 5
    df = _wrist_df(LEFT_WRIST_ID, x_values)
    wrist_speed = compute_wrist_speed(df, LEFT_WRIST_ID)

    strikes = detect_strikes_for_hand(
        wrist_speed, hand="left", min_interval_ms=1000.0, min_prominence=1.0
    )

    assert len(strikes) <= 1


def test_detect_strikes_for_hand_empty_when_all_nan():
    x_values = [math.nan] * 10
    df = _wrist_df(LEFT_WRIST_ID, x_values)
    wrist_speed = compute_wrist_speed(df, LEFT_WRIST_ID)

    strikes = detect_strikes_for_hand(wrist_speed, hand="left")

    assert strikes == []


def test_detect_strikes_combines_independent_hands():
    left_x = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.05,
        0.15,
        0.40,
        0.42,
        0.43,
        0.43,
        0.43,
        0.43,
    ]
    right_x = [0.5] * len(left_x)

    left_rows = _wrist_df(LEFT_WRIST_ID, left_x)
    right_rows = _wrist_df(RIGHT_WRIST_ID, right_x)
    df = pd.concat([left_rows, right_rows], ignore_index=True)

    strikes = detect_strikes(df, min_interval_ms=100.0, min_prominence=1.0)

    assert len(strikes) == 1
    assert strikes[0].hand == "left"


def test_detect_strikes_runs_without_crash_on_noisy_csv():
    x_values = []
    for i in range(200):
        if i % 37 == 0:
            x_values.append(math.nan)
        else:
            x_values.append(0.5 + 0.4 * math.sin(i / 5.0))
    df = _wrist_df(LEFT_WRIST_ID, x_values)

    strikes = detect_strikes(pd.concat([df, _wrist_df(RIGHT_WRIST_ID, [0.5] * 200)]))

    assert isinstance(strikes, list)
