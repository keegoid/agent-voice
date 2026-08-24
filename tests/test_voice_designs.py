from __future__ import annotations

from agent_voice.voices import VOICE_DESIGNS


def test_chesapeake_designs_lock_adult_gender_and_lower_register() -> None:
    assert VOICE_DESIGNS["chesapeake_balanced"] == (
        "A warm adult male British baritone, friendly and steady, not too formal, "
        "like someone "
        "who's right there with you. Clear, reassuring, but never stiff."
    )
    assert VOICE_DESIGNS["chesapeake_balanced_female"] == (
        "A warm adult female British contralto, friendly and steady, not too formal, "
        "like someone "
        "who's right there with you. Clear, reassuring, but never stiff."
    )
