import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from punch_analyzer.paths import CSV_OUTPUT_DIR, VIDEO_PATH, csv_path_for_video
from punch_analyzer.strike_detection import (
    DEFAULT_ACTIVITY_THRESHOLD,
    DEFAULT_CROSS_HAND_ARBITRATION_WINDOW_FRAMES,
    DEFAULT_DIRECTION_ANGLE_THRESHOLD_DEG,
    DEFAULT_EXTENSION_PEAK_PROMINENCE,
    DEFAULT_EXTENSION_RATIO_THRESHOLD,
    DEFAULT_GEOMETRIC_CONFIRM_WINDOW_FRAMES,
    DEFAULT_MIN_PEAK_SPEED,
    DEFAULT_MIN_SILENCE_MS,
    DEFAULT_MIN_VISIBILITY,
    DEFAULT_MIN_WINDOW_MS,
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_SAVGOL_WINDOW,
    ActivityWindow,
    Strike,
    detect_strikes,
    segment_activity_windows,
)


@dataclass
class ComboExpectation:
    name: str
    left: int
    right: int


@dataclass
class ComboScore:
    index: int
    name: str
    window: ActivityWindow
    expected_left: int
    expected_right: int
    detected_left: int
    detected_right: int
    confirmed_left: int
    confirmed_right: int
    strikes: list[Strike]  # tous les coups détectés dans la fenêtre, triés par frame_idx

    @property
    def left_match(self) -> bool:
        return self.confirmed_left == self.expected_left

    @property
    def right_match(self) -> bool:
        return self.confirmed_right == self.expected_right


def load_combo_list(path: Path) -> list[ComboExpectation]:
    """JSON attendu : liste ordonnée d'objets {"name": str, "left": int, "right": int},
    un par combo de la chorégraphie, dans l'ordre de tournage."""
    data = json.loads(path.read_text())
    return [
        ComboExpectation(name=item["name"], left=int(item["left"]), right=int(item["right"]))
        for item in data
    ]


def strikes_in_window(strikes: list[Strike], window: ActivityWindow) -> list[Strike]:
    return [s for s in strikes if window.start_frame <= s.frame_idx <= window.end_frame]


def filter_buffer_windows(windows: list[ActivityWindow], buffer_ms: float) -> list[ActivityWindow]:
    """Retire les fenêtres entièrement contenues dans la marge d'extraction avant
    start_ms (buffer_ms) -- lissage et détection ont tourné dessus, mais ce n'est
    pas du vrai contenu de combo, ne doit pas compter dans le score."""
    return [w for w in windows if w.end_ms > buffer_ms]


def filter_buffer_strikes(strikes: list[Strike], buffer_ms: float) -> list[Strike]:
    return [s for s in strikes if s.timestamp_ms >= buffer_ms]


@dataclass
class HandSpeedStats:
    hand: str
    n: int
    median: float
    minimum: float
    maximum: float
    stdev: float


def speed_distribution_by_hand(
    strikes: list[Strike], extension_threshold: float
) -> dict[str, list[float]]:
    """Vitesses au pic des coups confirmés par extension_ratio seul (>= extension_threshold),
    indépendamment de la vitesse -- sans ça, filtrer par geometric_pass introduirait un biais
    de survie (ne garderait que des vitesses déjà >= l'ancien seuil)."""
    by_hand: dict[str, list[float]] = {"left": [], "right": []}
    for s in strikes:
        if s.extension_ratio is not None and s.extension_ratio >= extension_threshold:
            by_hand[s.hand].append(s.speed)
    return by_hand


def compute_hand_speed_stats(speeds_by_hand: dict[str, list[float]]) -> dict[str, HandSpeedStats]:
    stats = {}
    for hand, speeds in speeds_by_hand.items():
        if len(speeds) < 2:
            continue
        stats[hand] = HandSpeedStats(
            hand=hand,
            n=len(speeds),
            median=statistics.median(speeds),
            minimum=min(speeds),
            maximum=max(speeds),
            stdev=statistics.stdev(speeds),
        )
    return stats


def is_speed_gap_net(stats: dict[str, HandSpeedStats], min_separation_ratio: float) -> bool:
    """"Net" si l'écart entre médianes gauche/droite est au moins min_separation_ratio
    fois le plus grand des deux écarts-types -- un écart d'un écart-type (ratio=1.0) ou
    plus n'est pas un bruit d'échantillonnage, distributions clairement séparées."""
    if "left" not in stats or "right" not in stats:
        return False
    gap = abs(stats["left"].median - stats["right"].median)
    spread = max(stats["left"].stdev, stats["right"].stdev)
    if spread == 0:
        return gap > 0
    return gap / spread >= min_separation_ratio


def calibrate_min_peak_speed_by_hand(stats: dict[str, HandSpeedStats]) -> dict[str, float]:
    """Seuil par main = médiane - 1 écart-type des coups confirmés par extension pour
    cette main, plancher à 0 -- laisse passer la grande majorité des vrais coups de
    cette main plutôt qu'un seuil unique calibré sur l'autre main."""
    return {hand: max(0.0, s.median - s.stdev) for hand, s in stats.items()}


def score_combos(
    windows: list[ActivityWindow],
    strikes: list[Strike],
    combos: list[ComboExpectation],
) -> list[ComboScore]:
    """Zippe fenêtres détectées et combos attendus dans l'ordre ; n'utilise que
    min(len(windows), len(combos)) paires -- appelant responsable de signaler
    tout désalignement de longueur avant d'utiliser ce résultat."""
    scores = []
    for i, (window, combo) in enumerate(zip(windows, combos)):
        window_strikes = strikes_in_window(strikes, window)
        left = [s for s in window_strikes if s.hand == "left"]
        right = [s for s in window_strikes if s.hand == "right"]
        scores.append(
            ComboScore(
                index=i + 1,
                name=combo.name,
                window=window,
                expected_left=combo.left,
                expected_right=combo.right,
                detected_left=len(left),
                detected_right=len(right),
                confirmed_left=sum(1 for s in left if s.geometric_pass),
                confirmed_right=sum(1 for s in right if s.geometric_pass),
                strikes=sorted(window_strikes, key=lambda s: s.frame_idx),
            )
        )
    return scores


def format_window_report(windows: list[ActivityWindow], buffer_ms: float = 0) -> list[str]:
    """Une ligne par fenêtre : début, fin, durée, silence depuis la fenêtre
    précédente -- pour voir à l'œil si des combos courts ont fusionné (silence
    trop court pour le seuil de coupure) ou disparu (durée jamais atteinte),
    avant de toucher aux seuils de segmentation. Les heures affichées sont
    relatives à start_ms (buffer_ms déjà retranché), pas au CSV brut."""
    lines = []
    previous_end_ms: float | None = None
    for i, window in enumerate(windows, start=1):
        duration_s = (window.end_ms - window.start_ms) / 1000
        if previous_end_ms is None:
            gap_str = "N/A"
        else:
            gap_str = f"{(window.start_ms - previous_end_ms) / 1000:.2f}s"
        lines.append(
            f"  [{i:>2}] {(window.start_ms - buffer_ms) / 1000:>6.2f}s -> "
            f"{(window.end_ms - buffer_ms) / 1000:>6.2f}s "
            f"(durée {duration_s:>5.2f}s, silence avant {gap_str})"
        )
        previous_end_ms = window.end_ms
    return lines


def print_report(scores: list[ComboScore], buffer_ms: float = 0) -> None:
    """Les heures affichées sont relatives à start_ms (buffer_ms déjà retranché)."""
    exact_matches = 0
    total_expected_left = total_expected_right = 0
    total_confirmed_left = total_confirmed_right = 0

    for score in scores:
        marker = "OK" if score.left_match and score.right_match else "!!"
        print(
            f"[{marker}] combo {score.index:>2} \"{score.name}\" "
            f"({(score.window.start_ms - buffer_ms) / 1000:.2f}s-"
            f"{(score.window.end_ms - buffer_ms) / 1000:.2f}s) : "
            f"gauche {score.confirmed_left}/{score.expected_left} confirmés "
            f"(détectés {score.detected_left}) -- "
            f"droite {score.confirmed_right}/{score.expected_right} confirmés "
            f"(détectés {score.detected_right})"
        )
        for strike in score.strikes:
            status = "confirmé" if strike.geometric_pass else "non confirmé"
            ext = f"{strike.extension_ratio:.3f}" if strike.extension_ratio is not None else "N/A"
            print(
                f"        [{strike.hand}] {(strike.timestamp_ms - buffer_ms) / 1000:.2f}s "
                f"pic vitesse={strike.speed:.3f} ext={ext} ({status})"
            )
        if score.left_match and score.right_match:
            exact_matches += 1
        total_expected_left += score.expected_left
        total_expected_right += score.expected_right
        total_confirmed_left += score.confirmed_left
        total_confirmed_right += score.confirmed_right

    print()
    print(
        f"{exact_matches}/{len(scores)} combos exacts (gauche+droite confirmés = attendu) -- "
        f"gauche {total_confirmed_left}/{total_expected_left} -- "
        f"droite {total_confirmed_right}/{total_expected_right}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare les coups détectés (confirmés par le filtre géométrique) "
        "aux combos attendus, combo par combo, en segmentant la vidéo en fenêtres d'activité."
    )
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--csv-dir", type=Path, default=CSV_OUTPUT_DIR)
    parser.add_argument(
        "--combos", type=Path, required=True,
        help="Chemin vers un JSON listant les combos attendus, voir load_combo_list().",
    )
    parser.add_argument(
        "--buffer-ms", type=float, default=0,
        help="Même valeur que --buffer-ms passé à landmark_extraction pour ce CSV -- "
        "fenêtres/coups dans la marge sont retirés du score, heures affichées recadrées.",
    )
    parser.add_argument("--activity-threshold", type=float, default=DEFAULT_ACTIVITY_THRESHOLD)
    parser.add_argument("--min-window-ms", type=float, default=DEFAULT_MIN_WINDOW_MS)
    parser.add_argument("--min-silence-ms", type=float, default=DEFAULT_MIN_SILENCE_MS)
    parser.add_argument("--extension-threshold", type=float, default=DEFAULT_EXTENSION_RATIO_THRESHOLD)
    parser.add_argument("--extension-prominence", type=float, default=DEFAULT_EXTENSION_PEAK_PROMINENCE)
    parser.add_argument("--angle-threshold", type=float, default=DEFAULT_DIRECTION_ANGLE_THRESHOLD_DEG)
    parser.add_argument("--min-peak-speed", type=float, default=DEFAULT_MIN_PEAK_SPEED)
    parser.add_argument("--savgol-window", type=int, default=DEFAULT_SAVGOL_WINDOW)
    parser.add_argument("--savgol-polyorder", type=int, default=DEFAULT_SAVGOL_POLYORDER)
    parser.add_argument("--min-visibility", type=float, default=DEFAULT_MIN_VISIBILITY)
    parser.add_argument(
        "--geometric-window-frames", type=int, default=DEFAULT_GEOMETRIC_CONFIRM_WINDOW_FRAMES
    )
    parser.add_argument(
        "--cross-hand-window", type=int, default=DEFAULT_CROSS_HAND_ARBITRATION_WINDOW_FRAMES
    )
    parser.add_argument(
        "--auto-calibrate-speed", action="store_true",
        help="Calcule la distribution de speed au pic par main (coups confirmés par "
        "extension_ratio seul) ; si l'écart gauche/droite est net, calibre min_peak_speed "
        "par main dessus (médiane - 1 écart-type) et relance la détection avant le score.",
    )
    parser.add_argument(
        "--speed-separation-ratio", type=float, default=1.0,
        help="Écart médiane/écart-type minimal (défaut 1.0 = un écart-type) pour juger "
        "l'écart gauche/droite 'net' avec --auto-calibrate-speed.",
    )
    args = parser.parse_args()

    csv_path = csv_path_for_video(args.video, args.csv_dir)
    if not csv_path.exists():
        print(f"CSV introuvable: {csv_path} (lancer landmark_extraction d'abord)")
        return

    df = pd.read_csv(csv_path)
    combos = load_combo_list(args.combos)

    windows = segment_activity_windows(
        df,
        activity_threshold=args.activity_threshold,
        min_window_ms=args.min_window_ms,
        min_silence_ms=args.min_silence_ms,
        min_visibility=args.min_visibility,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
    )
    strikes = detect_strikes(
        df,
        min_visibility=args.min_visibility,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        extension_threshold=args.extension_threshold,
        extension_prominence=args.extension_prominence,
        angle_threshold_deg=args.angle_threshold,
        min_peak_speed=args.min_peak_speed,
        geometric_window_frames=args.geometric_window_frames,
        cross_hand_window_frames=args.cross_hand_window,
    )

    if args.buffer_ms:
        dropped_windows = len(windows)
        windows = filter_buffer_windows(windows, args.buffer_ms)
        dropped_windows -= len(windows)
        dropped_strikes = len(strikes)
        strikes = filter_buffer_strikes(strikes, args.buffer_ms)
        dropped_strikes -= len(strikes)
        print(
            f"buffer_ms={args.buffer_ms:.0f} appliqué : {dropped_windows} fenêtre(s) et "
            f"{dropped_strikes} coup(s) dans la marge retirés avant score."
        )

    if args.auto_calibrate_speed:
        speeds_by_hand = speed_distribution_by_hand(strikes, args.extension_threshold)
        stats = compute_hand_speed_stats(speeds_by_hand)

        print("Distribution de speed au pic, coups confirmés par extension_ratio seul :")
        for hand in ("left", "right"):
            if hand in stats:
                s = stats[hand]
                print(
                    f"  {hand} (n={s.n}) : médiane={s.median:.3f} min={s.minimum:.3f} "
                    f"max={s.maximum:.3f} écart-type={s.stdev:.3f}"
                )
            else:
                n = len(speeds_by_hand.get(hand, []))
                print(f"  {hand} (n={n}) : pas assez de coups confirmés par extension pour un écart-type")

        if is_speed_gap_net(stats, args.speed_separation_ratio):
            gap = abs(stats["left"].median - stats["right"].median)
            spread = max(stats["left"].stdev, stats["right"].stdev)
            calibrated = calibrate_min_peak_speed_by_hand(stats)
            print(
                f"\nÉcart net : |médiane gauche - médiane droite| = {gap:.3f} >= "
                f"{args.speed_separation_ratio} x écart-type max ({spread:.3f}). "
                f"min_peak_speed calibré par main (médiane - 1 écart-type) :"
            )
            for hand, value in calibrated.items():
                print(f"  {hand} : {value:.3f} (au lieu de {args.min_peak_speed} global)")
            print()

            strikes = detect_strikes(
                df,
                min_visibility=args.min_visibility,
                savgol_window=args.savgol_window,
                savgol_polyorder=args.savgol_polyorder,
                extension_threshold=args.extension_threshold,
                extension_prominence=args.extension_prominence,
                angle_threshold_deg=args.angle_threshold,
                min_peak_speed=args.min_peak_speed,
                min_peak_speed_by_hand=calibrated,
                geometric_window_frames=args.geometric_window_frames,
                cross_hand_window_frames=args.cross_hand_window,
            )
            if args.buffer_ms:
                strikes = filter_buffer_strikes(strikes, args.buffer_ms)
        else:
            print(
                "\nÉcart gauche/droite non net (sous le seuil de séparation demandé) -- "
                "min_peak_speed reste global, rien de calibré.\n"
            )

    print(f"{len(windows)} fenêtres d'activité détectées, {len(combos)} combos attendus")
    for line in format_window_report(windows, args.buffer_ms):
        print(line)
    print()
    if len(windows) != len(combos):
        print(
            "ATTENTION : nombre de fenêtres != nombre de combos attendus -- "
            "désalignement probable (start_ms/end_ms mal calés, ou seuils de "
            "segmentation à ajuster). Le zip ci-dessous ne couvre que les "
            "premières paires communes, ne pas en tirer de conclusion fiable "
            "avant d'avoir corrigé cet écart."
        )
    print()

    scores = score_combos(windows, strikes, combos)
    print_report(scores, args.buffer_ms)


if __name__ == "__main__":
    main()
