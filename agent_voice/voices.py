"""Public voice-design presets."""

from __future__ import annotations

_COOL_FIGMENT_RAIN_VOICE_LOCKED = (
    "A young woman voice actor with a natural mid-range voice, guarded and sharp. "
    "She sounds like black-market tech that still works after the rain: cool, "
    "useful, unimpressed. Clipped phrasing, dry edge, neon-lit back-alley energy, "
    "clean studio tone. No low register, no gravel, no distortion; keep the "
    "scolding adult and sardonic. Maintain one consistent speaker identity for "
    "the entire output: same timbre, pitch center, age, accent, microphone "
    "distance, and vocal texture from sentence to sentence. Emotional swings "
    "should change only cadence, pause length, intensity, and emphasis. Do not "
    "morph into a different voice, character, register, accent, or age between "
    "sentences."
)

_CONSISTENT_SPEAKER_LOCK = (
    " Maintain one consistent speaker identity for the entire output: same timbre, "
    "pitch center, age, accent, microphone distance, and vocal texture from "
    "sentence to sentence. Emotional swings should change only cadence, pause "
    "length, intensity, and emphasis. Do not morph into a different voice, "
    "character, register, accent, or age between sentences."
)

_CYBERPUNK_COOL_MALE_VOICE_LOCKED = (
    "An adult man voice actor with a natural mid-baritone voice, guarded and "
    "sharp. He sounds like black-market tech that still works after the rain: "
    "cool, useful, unimpressed. Clipped phrasing, dry edge, neon-lit back-alley "
    "energy, clean studio tone. No gravel, no distortion, no theatrical growl; "
    "keep the delivery sardonic, controlled, and technically precise."
    + _CONSISTENT_SPEAKER_LOCK
)

_MALE_BOARDROOM_BARITONE_VOICE_LOCKED = (
    "An adult man voice actor with a composed executive baritone. Clear, dry, "
    "and boardroom-calm, with measured pacing and a low center of gravity. He "
    "sounds like a chief executive making a decision after reading the whole "
    "brief: restrained authority, direct diction, quiet pressure. Do not sound "
    "cyberpunk, sardonic, gravelly, theatrical, or radio-announcer polished."
    + _CONSISTENT_SPEAKER_LOCK
)

_MALE_MARKET_FLOOR_TENOR_VOICE_LOCKED = (
    "An adult man voice actor with a quick, focused market-floor tenor. Lean, "
    "alert, and analytical, with clipped financial-desk pacing and clean "
    "numbers-first articulation. He sounds like a trader calling risk in real "
    "time: fast enough to feel alive, controlled enough to be trusted. Do not "
    "sound cyberpunk, deep, raspy, salesman-like, or excitable."
    + _CONSISTENT_SPEAKER_LOCK
)

_MALE_EDITORIAL_BASS_VOICE_LOCKED = (
    "An adult man voice actor with a warm editorial bass. Thoughtful, grounded, "
    "and literate, with rounded vowels, clean breath support, and patient "
    "sentence endings. He sounds like a senior editor turning a messy draft into "
    "a sharp thesis: calm, humane, and exact. Do not sound cyberpunk, whispery, "
    "booming, sentimental, or sleepy."
    + _CONSISTENT_SPEAKER_LOCK
)

_MALE_MISSION_CONTROL_COMMANDER_VOICE_LOCKED = (
    "An adult man voice actor with a bright mission-control command voice. "
    "Medium pitch, crisp headset clarity, and decisive operational cadence. He "
    "sounds like a flight director calling the next maneuver: composed, alert, "
    "and visibly accountable. Do not sound cyberpunk, military-shouted, gravelly, "
    "mythic, or movie-trailer dramatic."
    + _CONSISTENT_SPEAKER_LOCK
)

_COOL_STREET_DEADPAN_VOICE_LOCKED = (
    "A young woman voice actor with a centered mid-range tone and street-smart "
    "disdain held under glass. Deadpan, clipped, and precise, with clean emphasis "
    "on the insult. Neon-lit back-alley energy without the growl: smooth studio "
    "voice, articulate consonants, restrained pressure. Do not sound low, raspy, "
    "cute, playful, or harsh."
    + _CONSISTENT_SPEAKER_LOCK
)

VOICE_DESIGNS: dict[str, str] = {
    "anime_genki": (
        "An adult woman voice actor with bright anime-heroine energy and quick, "
        "buoyant phrasing. Cheerful, expressive, high-focus enthusiasm, like she "
        "just found the critical clue and cannot wait to report it. Clear studio "
        "tone, crisp consonants, smiling lift on key words. Do not become shrill, "
        "childlike, breathless, or chaotic."
        + _CONSISTENT_SPEAKER_LOCK
    ),
    "anime_villain": (
        "An adult woman voice actor performing a calculating antagonist. "
        "Velvet-smooth, dangerously calm, and aristocratically precise. She "
        "savors important syllables, lets pauses land like chess moves, and keeps "
        "a quiet knowing smile under the line. She never needs volume; control is "
        "the pressure. Do not become cartoonish, cackling, raspy, or theatrical."
        + _CONSISTENT_SPEAKER_LOCK
    ),
    "cyberpunk_cool": _COOL_FIGMENT_RAIN_VOICE_LOCKED,
    "cyberpunk_cool_male": _CYBERPUNK_COOL_MALE_VOICE_LOCKED,
    "cool_figment_rain_voice_locked": _COOL_FIGMENT_RAIN_VOICE_LOCKED,
    "cool_street_deadpan_voice_locked": _COOL_STREET_DEADPAN_VOICE_LOCKED,
    "male_boardroom_baritone_locked": _MALE_BOARDROOM_BARITONE_VOICE_LOCKED,
    "male_market_floor_tenor_locked": _MALE_MARKET_FLOOR_TENOR_VOICE_LOCKED,
    "male_editorial_bass_locked": _MALE_EDITORIAL_BASS_VOICE_LOCKED,
    "male_mission_control_commander_locked": _MALE_MISSION_CONTROL_COMMANDER_VOICE_LOCKED,
    "peng_mythic": (
        "An adult woman voice actor performing an epic narrator in a fantasy saga. "
        "Clear, commanding, and resonant, building steadily like a gathering "
        "storm. She speaks with the certainty of someone who has seen the future "
        "unfold. Declarative statements land with finality, not as questions. "
        "Slow, deliberate pacing with conviction on sentence endings. Do not "
        "become raspy, ancient, monster-like, or melodramatic."
        + _CONSISTENT_SPEAKER_LOCK
    ),
    "anime_sultry": (
        "An adult woman voice actor with a smooth, intimate femme-fatale delivery. "
        "Low-to-mid pitch, deliberate pacing, close-mic warmth, and a controlled "
        "playful lilt at the edge of phrases. She sounds like she is sharing a "
        "dangerous secret and enjoying the leverage. Do not become breathless, "
        "childlike, sleepy, or whisper-only."
        + _CONSISTENT_SPEAKER_LOCK
    ),
    "anime_energetic": (
        "An adult woman voice actor delivering a dramatic anime declaration with "
        "building momentum. Clear and punchy at the start, then more intense "
        "through each phrase, with a lifted finish that stays controlled. "
        "Expressive, enthusiastic, and decisive. Do not become shrill, chaotic, "
        "childlike, or a different speaker when intensity rises."
        + _CONSISTENT_SPEAKER_LOCK
    ),
    "anime_whisper": (
        "An adult woman voice actor performing an intimate close-mic whisper. "
        "Soft, controlled, and emotionally near, with warm exhale texture and "
        "small pauses. Vulnerable and close without losing intelligibility. Do not "
        "turn into a child voice, pure breath, ASMR noise, or a different speaker "
        "between sentences."
        + _CONSISTENT_SPEAKER_LOCK
    ),
    "warm_wisdom": (
        "An adult woman voice actor with warm, thoughtful optimism. Mid-to-high "
        "pitch, gentle rounded intonation, brisk but calm pacing, and a smiling "
        "resonance throughout. She sounds like she is sharing a useful discovery "
        "with a friend: sincere, composed, and quietly delighted. Do not become "
        "saccharine, childlike, breathy, or motivational-speaker loud."
        + _CONSISTENT_SPEAKER_LOCK
    ),
    "sultry_commanding": (
        "An adult woman voice actor performing a powerful queen addressing her "
        "court. Low-to-mid, rich, and highly controlled. Each word lands with "
        "velvet authority; pauses imply that the room waits for her. Dark command "
        "on key words, elegant pressure, no hurry. Do not become raspy, booming, "
        "cartoon-villain theatrical, or breathless."
        + _CONSISTENT_SPEAKER_LOCK
    ),
}
