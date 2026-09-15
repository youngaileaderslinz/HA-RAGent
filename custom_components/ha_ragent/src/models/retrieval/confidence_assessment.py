from dataclasses import dataclass


@dataclass
class ConfidenceAssessment:
    level: str
    top_score: float = 0.0
    second_score: float = 0.0
    margin: float = 0.0
    ratio: float = 0.0
    agreeing_signals: tuple[str, ...] = ()
    disagreeing_signals: tuple[str, ...] = ()
    reason: str = "no candidates"
    candidate_scores: tuple[tuple[str, float], ...] = ()
    candidate_support: tuple[tuple[str, tuple[str, ...]], ...] = ()
    final_candidate_scores: tuple[tuple[str, float], ...] = ()
    continuity_boosts: tuple[tuple[str, float], ...] = ()
    absolute_strength: float = 0.0
    independent_signal_count: int = 0
