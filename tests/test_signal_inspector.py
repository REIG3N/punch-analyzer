from punch_analyzer.signal_inspector import (
    near_miss_speed_strikes,
    strikes_in_time_range,
)
from punch_analyzer.strike_detection import Strike


def _strike(hand: str, timestamp_ms: float, extension_ratio: float, speed: float) -> Strike:
    return Strike(
        hand=hand,
        frame_idx=int(timestamp_ms / 33.3),
        timestamp_ms=timestamp_ms,
        speed=speed,
        geometric_pass=extension_ratio >= 0.85 and speed >= 1.5,
        extension_ratio=extension_ratio,
    )


def test_strikes_in_time_range_filters_and_sorts_by_timestamp():
    strikes = [
        _strike("left", 5000.0, 0.9, 2.0),
        _strike("right", 500.0, 0.9, 2.0),
        _strike("left", 1500.0, 0.9, 2.0),
    ]

    result = strikes_in_time_range(strikes, 0.27, 1.81)

    assert [s.timestamp_ms for s in result] == [500.0, 1500.0]


def test_strikes_in_time_range_empty_when_nothing_matches():
    strikes = [_strike("left", 5000.0, 0.9, 2.0)]

    assert strikes_in_time_range(strikes, 0.27, 1.81) == []


def test_near_miss_speed_strikes_separates_near_perfect_from_below_speed():
    strikes = [
        _strike("left", 1000.0, 1.000, 1.427),  # près du seuil vitesse, extension parfaite
        _strike("right", 2000.0, 1.000, 1.6),  # extension parfaite, vitesse suffisante
        _strike("left", 3000.0, 0.5, 0.2),  # ni l'un ni l'autre
    ]

    near_perfect, below_speed = near_miss_speed_strikes(
        strikes, extension_floor=0.98, min_peak_speed=1.5
    )

    assert len(near_perfect) == 2
    assert [s.timestamp_ms for s in below_speed] == [1000.0]


def test_near_miss_speed_strikes_sorted_by_timestamp():
    strikes = [
        _strike("left", 3000.0, 1.0, 0.1),
        _strike("right", 1000.0, 1.0, 0.2),
    ]

    _near_perfect, below_speed = near_miss_speed_strikes(
        strikes, extension_floor=0.98, min_peak_speed=1.5
    )

    assert [s.timestamp_ms for s in below_speed] == [1000.0, 3000.0]
