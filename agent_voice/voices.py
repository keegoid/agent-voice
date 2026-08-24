"""Public voice-design presets."""

from __future__ import annotations

_CONSISTENT_SPEAKER = (
    " Maintain one consistent speaker identity for the entire output: same timbre, "
    "pitch center, age, accent, microphone distance, and vocal texture from "
    "sentence to sentence. Emotional swings should change only cadence, pause "
    "length, intensity, and emphasis. Do not morph into a different voice, "
    "character, register, accent, or age between sentences."
)

_CHESAPEAKE_BALANCED = (
    "A warm adult male British baritone, friendly and steady, not too formal, "
    "like someone "
    "who's right there with you. Clear, reassuring, but never stiff."
)

_CHESAPEAKE_BALANCED_FEMALE = (
    "A warm adult female British contralto, friendly and steady, not too formal, "
    "like someone "
    "who's right there with you. Clear, reassuring, but never stiff."
)

_ANIME_GENKI = (
    "An adult woman voice actor with bright anime-heroine energy and quick, "
    "buoyant phrasing. Cheerful, expressive, high-focus enthusiasm, like she "
    "just found the critical clue and cannot wait to report it. Clear studio "
    "tone, crisp consonants, smiling lift on key words. Do not become shrill, "
    "childlike, breathless, or chaotic."
    + _CONSISTENT_SPEAKER
)

_ANIME_VILLAIN = (
    "An adult woman voice actor performing a calculating antagonist. "
    "Velvet-smooth, dangerously calm, and aristocratically precise. She "
    "savors important syllables, lets pauses land like chess moves, and keeps "
    "a quiet knowing smile under the line. She never needs volume; control is "
    "the pressure. Do not become cartoonish, cackling, raspy, or theatrical."
    + _CONSISTENT_SPEAKER
)

_COOL_FIGMENT_RAIN = (
    "A young woman voice actor with a natural mid-range voice, guarded and "
    "sharp. She sounds like black-market tech that still works after the rain: "
    "cool, useful, unimpressed. Clipped phrasing, dry edge, neon-lit back-alley "
    "energy, clean studio tone. Spoken target: soft-spoken and close to the "
    "microphone, with subtle ASMR texture focused on delicate consonant detail "
    "and tiny dry pauses. Use only a little breath at phrase endings; keep the "
    "words crisp and clean. Keep the pitch natural and lightly lifted; do not "
    "make the voice deeper to sound quiet. The result should feel almost "
    "whispered without losing the original speaking voice. No low register, no "
    "gravel, no distortion, no mouth-noise performance, no sleepy murmur. Keep "
    "the delivery scolding adult and sardonic. Maintain one consistent speaker "
    "identity for the entire output: same timbre, pitch center, age, accent, "
    "microphone distance, and vocal texture from sentence to sentence. Emotional "
    "swings should change only cadence, pause length, intensity, and emphasis. "
    "Do not morph into a different voice, character, register, accent, or age "
    "between sentences."
)

_CYBERPUNK_COOL = _COOL_FIGMENT_RAIN

_CYBERPUNK_COOL_MALE = (
    "An adult man voice actor with a natural mid-baritone voice, guarded and "
    "sharp. He sounds like black-market tech that still works after the rain: "
    "cool, useful, unimpressed. Clipped phrasing, dry edge, neon-lit back-alley "
    "energy, clean studio tone. No gravel, no distortion, no theatrical growl; "
    "keep the delivery sardonic, controlled, and technically precise."
    + _CONSISTENT_SPEAKER
)

_MALE_BOARDROOM_BARITONE = (
    "An adult man voice actor with a composed executive baritone. Clear, dry, "
    "and boardroom-calm, with measured pacing and a low center of gravity. He "
    "sounds like a chief executive making a decision after reading the whole "
    "brief: restrained authority, direct diction, quiet pressure. Do not sound "
    "cyberpunk, sardonic, gravelly, theatrical, or radio-announcer polished."
    + _CONSISTENT_SPEAKER
)

_MALE_MARKET_FLOOR_TENOR = (
    "An adult man voice actor with a quick, focused market-floor tenor. Lean, "
    "alert, and analytical, with clipped financial-desk pacing and clean "
    "numbers-first articulation. He sounds like a trader calling risk in real "
    "time: fast enough to feel alive, controlled enough to be trusted. Do not "
    "sound cyberpunk, deep, raspy, salesman-like, or excitable."
    + _CONSISTENT_SPEAKER
)

_MALE_EDITORIAL_BASS = (
    "An adult man voice actor with a warm editorial bass. Thoughtful, grounded, "
    "and literate, with rounded vowels, clean breath support, and patient "
    "sentence endings. He sounds like a senior editor turning a messy draft into "
    "a sharp thesis: calm, humane, and exact. Do not sound cyberpunk, whispery, "
    "booming, sentimental, or sleepy."
    + _CONSISTENT_SPEAKER
)

_MALE_MISSION_CONTROL_COMMANDER = (
    "An adult man voice actor with a bright mission-control command voice. "
    "Medium pitch, crisp headset clarity, and decisive operational cadence. He "
    "sounds like a flight director calling the next maneuver: composed, alert, "
    "and visibly accountable. Do not sound cyberpunk, military-shouted, gravelly, "
    "mythic, or movie-trailer dramatic."
    + _CONSISTENT_SPEAKER
)

_COOL_STREET_DEADPAN = (
    "A young woman voice actor with a mid-range tone and dry deadpan confidence. "
    "Street-smart, neon-lit back-alley cool under pressure. "
    "Not bubbly and no high-pitched or fast-paced genki."
    + _CONSISTENT_SPEAKER
)

_PENG_MYTHIC = (
    "An adult woman voice actor performing an epic narrator in a fantasy saga. "
    "Clear, commanding, and resonant, building steadily like a gathering "
    "storm. She speaks with the certainty of someone who has seen the future "
    "unfold. Declarative statements land with finality, not as questions. "
    "Slow, deliberate pacing with conviction on sentence endings. Do not "
    "become raspy, ancient, monster-like, or melodramatic."
    + _CONSISTENT_SPEAKER
)

_QUESTLINE_DEADPAN = (
    "An adult woman voice actor with a slightly brighter pitch center, "
    "delivering quiet progress cues like a mythic quest update from a neon back "
    "alley. Energetic whisper, clipped timing, dry deadpan confidence, and a "
    "subtle thrill when the path opens. She sounds like black-market tech left "
    "in the rain, still working, and pleased about it. Crisp diction, close-mic "
    "focus, bright decisive finishes."
    + _CONSISTENT_SPEAKER
)

_ANIME_SULTRY = (
    "An adult woman voice actor with a smooth, intimate femme-fatale delivery. "
    "Low-to-mid pitch, deliberate pacing, close-mic warmth, and a controlled "
    "playful lilt at the edge of phrases. She sounds like she is sharing a "
    "dangerous secret and enjoying the leverage. Do not become breathless, "
    "childlike, sleepy, or whisper-only."
    + _CONSISTENT_SPEAKER
)

_ANIME_ENERGETIC = (
    "An adult woman voice actor delivering a dramatic anime declaration with "
    "building momentum. Clear and punchy at the start, then more intense "
    "through each phrase, with a lifted finish that stays controlled. "
    "Expressive, enthusiastic, and decisive. Do not become shrill, chaotic, "
    "childlike, or a different speaker when intensity rises."
    + _CONSISTENT_SPEAKER
)

_ANIME_WHISPER = (
    "An adult woman voice actor performing an intimate close-mic whisper. "
    "Soft, controlled, and emotionally near, with warm exhale texture and "
    "small pauses. Vulnerable and close without losing intelligibility. Do not "
    "turn into a child voice, pure breath, ASMR noise, or a different speaker "
    "between sentences."
    + _CONSISTENT_SPEAKER
)

_WARM_WISDOM = (
    "An adult woman voice actor with warm, thoughtful optimism. Mid-to-high "
    "pitch, gentle rounded intonation, brisk but calm pacing, and a smiling "
    "resonance throughout. She sounds like she is sharing a useful discovery "
    "with a friend: sincere, composed, and quietly delighted. Do not become "
    "saccharine, childlike, breathy, or motivational-speaker loud."
    + _CONSISTENT_SPEAKER
)

_SULTRY_COMMANDING = (
    "An adult woman voice actor performing a powerful queen addressing her "
    "court. Low-to-mid, rich, and highly controlled. Each word lands with "
    "velvet authority; pauses imply that the room waits for her. Dark command "
    "on key words, elegant pressure, no hurry. Do not become raspy, booming, "
    "cartoon-villain theatrical, or breathless."
    + _CONSISTENT_SPEAKER
)

VOICE_DESIGNS: dict[str, str] = {
    "chesapeake_balanced": _CHESAPEAKE_BALANCED,
    "chesapeake_balanced_female": _CHESAPEAKE_BALANCED_FEMALE,
    "anime_genki": _ANIME_GENKI,
    "anime_villain": _ANIME_VILLAIN,
    "cyberpunk_cool": _CYBERPUNK_COOL,
    "cyberpunk_cool_male": _CYBERPUNK_COOL_MALE,
    "cool_figment_rain": _COOL_FIGMENT_RAIN,
    "cool_street_deadpan": _COOL_STREET_DEADPAN,
    "male_boardroom_baritone": _MALE_BOARDROOM_BARITONE,
    "male_market_floor_tenor": _MALE_MARKET_FLOOR_TENOR,
    "male_editorial_bass": _MALE_EDITORIAL_BASS,
    "male_mission_control_commander": _MALE_MISSION_CONTROL_COMMANDER,
    "peng_mythic": _PENG_MYTHIC,
    "questline_deadpan": _QUESTLINE_DEADPAN,
    "anime_sultry": _ANIME_SULTRY,
    "anime_energetic": _ANIME_ENERGETIC,
    "anime_whisper": _ANIME_WHISPER,
    "warm_wisdom": _WARM_WISDOM,
    "sultry_commanding": _SULTRY_COMMANDING,
}
