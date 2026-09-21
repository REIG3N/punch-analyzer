import math

import pandas as pd
import pytest

from punch_analyzer.strike_detection import (
    ARM_LANDMARKS,
    LEFT_HIP_ID,
    LEFT_SHOULDER_ID,
    LEFT_WRIST_ID,
    RIGHT_HIP_ID,
    RIGHT_SHOULDER_ID,
    RIGHT_WRIST_ID,
    Strike,
    _arbitrate_cross_hand,
    _fill_short_gaps,
    compute_arm_geometry,
    compute_torso_center_velocity,
    compute_wrist_speed,
    detect_strikes,
    detect_strikes_for_hand,
    pivot_landmarks,
    segment_activity_windows,
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


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 3 * t**2 - 2 * t**3


def _lerp(a: float, b: float, s: float) -> float:
    return a + (b - a) * s


def _arm_pose(
    shoulder: tuple[float, float],
    upper_arm_angle_deg: float,
    forearm_angle_deg: float,
    upper_arm_len: float = 0.2,
    forearm_len: float = 0.2,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bras à longueurs de segments fixes (réaliste), seul l'angle de l'avant-bras
    varie pour simuler l'extension : garantit une extension_ratio monotone le long
    du mouvement (contrairement à une interpolation linéaire brute des points, qui
    peut créer des pics d'extension parasites en cours de trajet)."""
    theta1 = math.radians(upper_arm_angle_deg)
    elbow = (
        shoulder[0] + upper_arm_len * math.cos(theta1),
        shoulder[1] + upper_arm_len * math.sin(theta1),
    )
    theta2 = math.radians(forearm_angle_deg)
    wrist = (elbow[0] + forearm_len * math.cos(theta2), elbow[1] + forearm_len * math.sin(theta2))
    return elbow, wrist


BENT_DELTA_DEG = 100.0
EXTENDED_DELTA_DEG = 0.0


def _punch_cycle_df(
    hand: str,
    upper_arm_angle_deg: float = 0.0,
    bent_delta_deg: float = BENT_DELTA_DEG,
    extended_delta_deg: float = EXTENDED_DELTA_DEG,
    shoulder: tuple[float, float] = (0.0, 0.0),
    other_shoulder: tuple[float, float] = (0.3, 0.0),
    hip: tuple[float, float] = (0.0, 0.5),
    hold_frames: int = 5,
    transition_frames: int = 10,
    visibility: float = 0.9,
) -> pd.DataFrame:
    """Simule un cycle garde -> extension -> garde. Le trajet aller/retour utilise un
    smoothstep : vitesse nulle en garde, pic de vitesse en cours de trajet, vitesse
    quasi nulle au point d'extension max (arrêt bref à l'impact, comme un vrai coup)."""
    ids = ARM_LANDMARKS[hand]
    other_shoulder_id = RIGHT_SHOULDER_ID if hand == "left" else LEFT_SHOULDER_ID

    def pose(delta_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
        return _arm_pose(shoulder, upper_arm_angle_deg, upper_arm_angle_deg + delta_deg)

    positions: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for _ in range(hold_frames):
        positions.append(pose(bent_delta_deg))
    for i in range(transition_frames):
        s = _smoothstep(i / (transition_frames - 1))
        positions.append(pose(_lerp(bent_delta_deg, extended_delta_deg, s)))
    for i in range(transition_frames):
        s = _smoothstep(i / (transition_frames - 1))
        positions.append(pose(_lerp(extended_delta_deg, bent_delta_deg, s)))
    for _ in range(hold_frames):
        positions.append(pose(bent_delta_deg))

    rows = []
    for frame_idx, (elbow, wrist) in enumerate(positions):
        t_ms = frame_idx * FRAME_MS
        landmarks = {
            ids["shoulder"]: shoulder,
            ids["elbow"]: elbow,
            ids["wrist"]: wrist,
            other_shoulder_id: other_shoulder,
            LEFT_HIP_ID: hip,
            RIGHT_HIP_ID: (hip[0] + other_shoulder[0], hip[1]),
        }
        for landmark_id, (x, y) in landmarks.items():
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_ms": t_ms,
                    "landmark_id": landmark_id,
                    "x": x,
                    "y": y,
                    "z": 0.0,
                    "visibility": visibility,
                }
            )
    return pd.DataFrame(rows)


def _jab_cycle_df(**kwargs) -> pd.DataFrame:
    return _punch_cycle_df("left", **kwargs)


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


def test_compute_wrist_speed_ignores_low_visibility_points():
    df = _wrist_df(LEFT_WRIST_ID, [0.2] * 5 + [0.9] * 5)
    df.loc[5:6, "visibility"] = 0.1  # confiance basse mais pas NaN

    speed = compute_wrist_speed(df, LEFT_WRIST_ID, min_visibility=0.5, max_gap_frames=5)

    # Les points basse-confiance sont masqués comme un trou : pas d'explosion de vitesse.
    assert speed["speed"].dropna().le(50).all()


def test_compute_wrist_speed_subtracts_torso_velocity():
    n = 10
    wrist_x = [0.5 + 0.1 * i for i in range(n)]
    shoulder_x = [0.5 + 0.1 * i for i in range(n)]  # tronc bouge exactement comme le poignet

    rows = []
    for frame_idx in range(n):
        t_ms = frame_idx * FRAME_MS
        for landmark_id, x in (
            (LEFT_WRIST_ID, wrist_x[frame_idx]),
            (LEFT_SHOULDER_ID, shoulder_x[frame_idx]),
            (RIGHT_SHOULDER_ID, shoulder_x[frame_idx]),
        ):
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_ms": t_ms,
                    "landmark_id": landmark_id,
                    "x": x,
                    "y": 0.5,
                    "z": 0.0,
                    "visibility": 0.9,
                }
            )
    df = pd.DataFrame(rows)
    wide = pivot_landmarks(df)
    torso_velocity = compute_torso_center_velocity(wide, savgol_window=3, savgol_polyorder=1)

    speed = compute_wrist_speed(
        df, LEFT_WRIST_ID, torso_velocity, savgol_window=3, savgol_polyorder=1
    )

    # Le poignet bouge uniquement parce que le tronc bouge : vitesse relative ~0.
    assert speed["speed"].dropna().le(0.05).all()


def test_detect_strikes_for_hand_extension_peak_not_double_counted():
    df = _jab_cycle_df()
    wide = pivot_landmarks(df)
    torso_velocity = compute_torso_center_velocity(wide)
    wrist_speed = compute_wrist_speed(df, LEFT_WRIST_ID, torso_velocity)
    geometry = compute_arm_geometry(wide, "left")

    strikes = detect_strikes_for_hand(geometry, wrist_speed, "left", min_peak_speed=0.0)

    assert len(strikes) == 1


def test_detect_strikes_for_hand_geometric_pass_true_for_full_extension():
    df = _jab_cycle_df()
    wide = pivot_landmarks(df)
    torso_velocity = compute_torso_center_velocity(wide)
    wrist_speed = compute_wrist_speed(df, LEFT_WRIST_ID, torso_velocity)
    geometry = compute_arm_geometry(wide, "left")

    strikes = detect_strikes_for_hand(geometry, wrist_speed, "left", min_peak_speed=0.0)

    assert strikes[0].geometric_pass is True
    assert strikes[0].extension_ratio == pytest.approx(1.0, abs=0.05)


def test_detect_strikes_for_hand_rejected_when_speed_too_low():
    # Même trajectoire d'extension, mais étalée sur beaucoup plus de frames :
    # la vitesse de pointe ne dépasse jamais le seuil minimal requis.
    df = _jab_cycle_df(transition_frames=200)
    wide = pivot_landmarks(df)
    torso_velocity = compute_torso_center_velocity(wide)
    wrist_speed = compute_wrist_speed(df, LEFT_WRIST_ID, torso_velocity)
    geometry = compute_arm_geometry(wide, "left")

    strikes = detect_strikes_for_hand(
        geometry, wrist_speed, "left", min_peak_speed=1.5, extension_prominence=0.05
    )

    assert len(strikes) == 1
    assert strikes[0].geometric_pass is False


def test_detect_strikes_for_hand_rejected_when_extension_incomplete():
    # Un "crochet" : le bras se déplie partiellement mais ne s'aligne jamais
    # (extension_ratio max ~0.77, sous le seuil de 0.85).
    df = _punch_cycle_df("left", extended_delta_deg=80.0)
    wide = pivot_landmarks(df)
    torso_velocity = compute_torso_center_velocity(wide)
    wrist_speed = compute_wrist_speed(df, LEFT_WRIST_ID, torso_velocity)
    geometry = compute_arm_geometry(wide, "left")

    strikes = detect_strikes_for_hand(
        geometry, wrist_speed, "left", min_peak_speed=0.0, extension_prominence=0.05
    )

    assert len(strikes) == 1
    assert strikes[0].geometric_pass is False
    assert strikes[0].extension_ratio < 0.85


def test_detect_strikes_for_hand_rejected_when_wrist_drops_below_head_band():
    # Bras tendu mais vers le bas (mouvement parasite), pas vers la tête.
    df = _punch_cycle_df("left", upper_arm_angle_deg=90.0)
    wide = pivot_landmarks(df)
    torso_velocity = compute_torso_center_velocity(wide)
    wrist_speed = compute_wrist_speed(df, LEFT_WRIST_ID, torso_velocity)
    geometry = compute_arm_geometry(wide, "left")

    strikes = detect_strikes_for_hand(
        geometry, wrist_speed, "left", min_peak_speed=0.0, extension_prominence=0.05
    )

    assert len(strikes) == 1
    assert strikes[0].geometric_pass is False


def test_detect_strikes_for_hand_empty_when_all_nan():
    x_values = [math.nan] * 10
    df = _wrist_df(LEFT_WRIST_ID, x_values)
    wrist_speed = compute_wrist_speed(df, LEFT_WRIST_ID)
    geometry = pd.DataFrame(
        columns=["frame_idx", "timestamp_ms", "extension_ratio", "wrist_y", "shoulder_y", "torso_length", "angle_deg"]
    )

    strikes = detect_strikes_for_hand(geometry, wrist_speed, "left")

    assert strikes == []


def test_detect_strikes_combines_independent_hands():
    left_df = _jab_cycle_df()
    # Main droite : aucune donnée -> aucun coup détecté côté droit, pas de crash.
    right_rows = []
    for frame_idx in left_df["frame_idx"].unique():
        for landmark_id in (RIGHT_WRIST_ID,):
            right_rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_ms": frame_idx * FRAME_MS,
                    "landmark_id": landmark_id,
                    "x": 0.5,
                    "y": 0.5,
                    "z": 0.0,
                    "visibility": 0.9,
                }
            )
    df = pd.concat([left_df, pd.DataFrame(right_rows)], ignore_index=True)

    strikes = detect_strikes(df, min_peak_speed=0.0)

    assert len(strikes) == 1
    assert strikes[0].hand == "left"


def test_detect_strikes_runs_without_crash_on_noisy_csv():
    df = _jab_cycle_df()
    strikes = detect_strikes(df)
    assert isinstance(strikes, list)


def test_arbitrate_cross_hand_keeps_only_higher_extension_when_simultaneous():
    strikes = [
        Strike(hand="left", frame_idx=100, timestamp_ms=3333.0, speed=2.0, geometric_pass=True, extension_ratio=0.90),
        Strike(hand="right", frame_idx=102, timestamp_ms=3400.0, speed=2.0, geometric_pass=True, extension_ratio=0.97),
    ]

    _arbitrate_cross_hand(strikes, window_frames=5)

    assert strikes[0].geometric_pass is False
    assert strikes[1].geometric_pass is True


def test_arbitrate_cross_hand_leaves_distant_strikes_untouched():
    strikes = [
        Strike(hand="left", frame_idx=10, timestamp_ms=333.0, speed=2.0, geometric_pass=True, extension_ratio=0.90),
        Strike(hand="right", frame_idx=200, timestamp_ms=6666.0, speed=2.0, geometric_pass=True, extension_ratio=0.97),
    ]

    _arbitrate_cross_hand(strikes, window_frames=5)

    assert strikes[0].geometric_pass is True
    assert strikes[1].geometric_pass is True


def test_arbitrate_cross_hand_ignores_unconfirmed_candidates():
    strikes = [
        Strike(hand="left", frame_idx=100, timestamp_ms=3333.0, speed=2.0, geometric_pass=True, extension_ratio=0.90),
        Strike(hand="right", frame_idx=102, timestamp_ms=3400.0, speed=0.1, geometric_pass=False, extension_ratio=0.97),
    ]

    _arbitrate_cross_hand(strikes, window_frames=5)

    assert strikes[0].geometric_pass is True
    assert strikes[1].geometric_pass is False


def _speed_df(x_values: list[float], landmark_id: int = LEFT_WRIST_ID) -> pd.DataFrame:
    rows = []
    for frame_idx, x in enumerate(x_values):
        t_ms = frame_idx * FRAME_MS
        for lid, lx in ((landmark_id, x), (LEFT_SHOULDER_ID, 0.0), (RIGHT_SHOULDER_ID, 0.3)):
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_ms": t_ms,
                    "landmark_id": lid,
                    "x": lx,
                    "y": 0.5,
                    "z": 0.0,
                    "visibility": 0.9,
                }
            )
    return pd.DataFrame(rows)


def _moving(n: int, step: float = 0.02, start: float = 0.0) -> list[float]:
    return [start + step * i for i in range(n)]


def _still(n: int, value: float = 0.0) -> list[float]:
    return [value] * n


def test_segment_activity_windows_splits_on_long_silence():
    x_values = _moving(15) + _still(90, 0.28) + _moving(15, start=0.28)
    df = _speed_df(x_values)

    windows = segment_activity_windows(df, savgol_window=3, savgol_polyorder=1)

    assert len(windows) == 2


def test_segment_activity_windows_bridges_brief_lull_inside_combo():
    x_values = _moving(10) + _still(5, 0.2) + _moving(15, start=0.2)
    df = _speed_df(x_values)

    windows = segment_activity_windows(df, savgol_window=3, savgol_polyorder=1)

    assert len(windows) == 1


def test_segment_activity_windows_drops_short_blip():
    x_values = _still(40, 0.0) + _moving(3) + _still(40, 0.06)
    df = _speed_df(x_values)

    windows = segment_activity_windows(df, savgol_window=3, savgol_polyorder=1)

    assert windows == []
