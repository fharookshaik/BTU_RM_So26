"""Algorithm 1 from Wang et al. 2022: raw label → 7-class taxonomy."""

SUBSTRINGS = [
    ("silence", "silence"), ("pre-chorus", "verse"), ("prechorus", "verse"),
    ("refrain", "chorus"), ("chorus", "chorus"), ("theme", "chorus"),
    ("stutter", "chorus"), ("verse", "verse"), ("rap", "verse"),
    ("section", "verse"), ("slow", "verse"), ("build", "verse"),
    ("dialog", "verse"), ("intro", "intro"), ("raden", "intro"),
    ("opening", "intro"), ("bridge", "bridge"), ("trans", "bridge"),
    ("out", "outro"), ("coda", "outro"), ("ending", "outro"),
    ("break", "inst"), ("inst", "inst"), ("interlude", "inst"),
    ("improv", "inst"), ("solo", "inst"),
]

CLASSES = ["intro", "verse", "chorus", "bridge", "outro", "inst", "silence"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def convert_label(label: str) -> str:
    """Map a raw segment label to one of the 7 classes (or 'end')."""
    if label.strip().lower() == "end":
        return "end"
    for s1, s2 in SUBSTRINGS:
        if s1 in label.lower():
            return s2
    return "inst"


def convert_segments(boundaries, labels):
    """Convert a list of (boundary, raw_label) pairs to converted labels.

    Returns:
        converted: list of str — converted label per segment
        token_seq: list of int — index in CLASSES for segments that are not 'end'
    """
    converted = []
    token_seq = []
    for b, lbl in zip(boundaries, labels):
        c = convert_label(lbl)
        converted.append(c)
        if c != "end" and c in CLASS_TO_IDX:
            token_seq.append(CLASS_TO_IDX[c])
    return converted, token_seq
