"""K-R6 decision rule and expansion prompts (docs/confirm_protocol.md)."""

from __future__ import annotations

from ragkit.confirmbench import BeirResult, Row, decide, expansion_system
from ragkit.vocabbench import EXPANSION_SYSTEM, METRICS, Result


def _row(scifact: float, nfcorpus: float, eu_mrr: float, pack: float, wrong: int) -> Row:
    n = 40
    eu = Result({m: [eu_mrr] * n for m in METRICS}, [i < wrong for i in range(n)])
    eu.scores["pack"] = [pack] * n
    beir = {
        "scifact": BeirResult([scifact + (i % 2) * 0.01 for i in range(n)], [1.0] * n),
        "nfcorpus": BeirResult([nfcorpus + (i % 3) * 0.01 for i in range(n)], [1.0] * n),
    }
    return Row(beir, eu, Result({m: [1.0] for m in METRICS}, [False]))


BASE = _row(0.6, 0.3, 0.7, 0.9, 2)


def test_clear_gain_everywhere_is_adopted() -> None:
    assert decide(BASE, _row(0.65, 0.35, 0.75, 0.9, 2)).status == "adopted"


def test_one_set_below_zero_blocks_adoption_despite_pooled_gain() -> None:
    verdict = decide(BASE, _row(0.80, 0.29, 0.75, 0.9, 2))
    assert verdict.d_pooled[1] > 0
    assert verdict.status == "not adopted (no evidence)"
    assert any("nfcorpus" in f for f in verdict.failed)


def test_eu_serving_context_blocks_adoption() -> None:
    assert decide(BASE, _row(0.65, 0.35, 0.75, 0.9, 3)).status != "adopted"
    assert decide(BASE, _row(0.65, 0.35, 0.75, 0.8, 2)).status != "adopted"


def test_pooled_loss_is_rejected() -> None:
    assert decide(BASE, _row(0.5, 0.2, 0.75, 0.9, 2)).status == "rejected"


def test_eu_prompt_is_k_r5_prompt() -> None:
    # The synthetic passages are replayed from the K-R5 cache, keyed by the exact prompt.
    assert expansion_system("eu") == expansion_system("synthetic") == EXPANSION_SYSTEM
