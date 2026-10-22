"""Real-model checks for the model signals: need the [st] extra and the Hugging Face Hub."""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from ragkit.modelsignals import CrossEncoderScorer, SquadNullScorer

pytestmark = [pytest.mark.slow, pytest.mark.network]

_QUESTION = "Which body approves the capacity calculation methodology?"
_ANSWERING = (
    "All regulatory authorities of the capacity calculation region approve the methodology."
)
_UNRELATED = "Hydropower plants convert the kinetic energy of falling water into electricity."


def test_cross_encoder_ranks_relevant_passage_higher() -> None:
    relevant, unrelated = CrossEncoderScorer().score(_QUESTION, [_ANSWERING, _UNRELATED])
    assert relevant > unrelated


def test_squad2_reader_prefers_span_when_the_answer_is_stated() -> None:
    answering, unrelated = SquadNullScorer().score(_QUESTION, [_ANSWERING, _UNRELATED])
    assert answering > 0.0
    assert answering > unrelated
