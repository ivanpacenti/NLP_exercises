from fastapi import FastAPI
from pydantic import BaseModel
import re
from typing import List, Tuple, Optional

app = FastAPI()


class TextInput(BaseModel):
    text: str


# Tokenizer: include lettere danesi, apostrofi ASCII+unicode, e trattini
WORD_RE = re.compile(r"[a-zA-ZæøåÆØÅ'’\-]+")

# --- Lexicons / phrases ------------------------------------------------------

POS_PHRASES_RAW = [
    "well structured", "well-structured", "well organized", "well-organized",
    "learned a lot", "lærte virkelig meget", "jeg lærte virkelig meget",
    "super actionable", "hands-on", "inspiring", "beautifully presented",
    "mega godt kursus", "rigtig god struktur",
    "great course",
    "clear structure",
    "useful exercises",
    "practical and fun",
    "the teacher was nice",
    "nice and helpful",
    "loved the project",
    "loved the project work",
    "iterative feedback",
    "feedback-loopet",
    "jeg lærte meget",
    "var god til at forklare",
    "bandt det hele sammen",
    "gav mening",
    "tydelig formidling",
    "stærke diskussioner",
]

NEG_PHRASES_RAW = [
    "did not learn much", "didn't learn much",
    "waste of time", "spild af tid",
    "underprepared", "too many gaps", "rodet", "frustrerende",
    "uklare krav", "for få forklaringer", "confusing",
]

NEU_PHRASES_RAW = [
    "overall fine", "nothing special", "okay", "ok",
    "helt fint", "ikke noget wow", "fine, nothing special",
]

NEGATION_WORDS = {"not", "no", "ikke", "ingen", "aldrig", "never"}

POS_WORDS_RAW = {
    "great", "excellent", "amazing", "useful", "clear", "interesting", "engaging",
    "helpful", "motivating", "inspiring", "fantastic", "actionable",
    "god", "godt", "gode", "fantastisk", "fremragende", "lærerigt", "tydelig",
    "spændende", "interessant", "anbefaler", "engageret",
    "liked", "love", "loved", "nice", "learned", "tools",
    "sharp", "quick", "fast", "concrete", "clearly", "solid", "practical", "fun",
    "energetic", "mega", "hjælpsom", "præcist", "elegant", "velvalgte",
    "elskede", "lærte", "tryg", "stærk", "mening",
}

NEG_WORDS_RAW = {
    "bad", "boring", "dry", "confusing", "unclear", "terrible", "awful",
    "useless", "unhelpful", "disorganized", "messy", "frustrating",
    "dårlig", "kedelig", "tørt", "forvirrende", "uklar", "elendig", "uorganiseret",
    "rodet", "frustrerende",
}

MIX_MARKERS = {"but", "however", "though", "wish", "could", "men", "dog", "ønsker", "kunne"}

INTENSIFIERS = {"very", "really", "extremely", "meget", "virkelig", "mega", "super"}


def normalize_space(s: str) -> str:
    # normalizza whitespace e apostrofi unicode -> ascii per consistenza
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Dedup + lowercase all at import time (avoid double-counting)
POS_PHRASES = sorted(set(normalize_space(p.lower()) for p in POS_PHRASES_RAW))
NEG_PHRASES = sorted(set(normalize_space(p.lower()) for p in NEG_PHRASES_RAW))
NEU_PHRASES = sorted(set(normalize_space(p.lower()) for p in NEU_PHRASES_RAW))

POS_WORDS = set(w.lower().replace("’", "'") for w in POS_WORDS_RAW)
NEG_WORDS = set(w.lower().replace("’", "'") for w in NEG_WORDS_RAW)


# --- Helpers -----------------------------------------------------------------

def tokenize(s: str) -> List[str]:
    s = normalize_space(s.lower())
    return WORD_RE.findall(s)


def quantize_label(raw_score: float) -> int:
    # -3 negative, 0 neutral, 3 positive
    if raw_score <= -1:
        return -3
    if raw_score >= 1:
        return 3
    return 0


def phrase_hit(phrase: str, s: str) -> bool:
    """
    More robust phrase match:
    - if phrase contains hyphen/apostrophe, substring is acceptable
    - else use word-boundary regex to avoid accidental matches
    """
    if any(ch in phrase for ch in "-'"):
        return phrase in s
    return re.search(rf"\b{re.escape(phrase)}\b", s) is not None


def split_on_contrast(s: str) -> Optional[Tuple[str, str]]:
    """
    Split on common contrast markers (but/men/however/dog).
    Keep it simple and low-cost.
    """
    # space-padded to reduce false hits inside words
    for m in [" but ", " however ", " men ", " dog "]:
        if m in s:
            a, b = s.split(m, 1)
            return a.strip(), b.strip()
    return None


def score_segment(text: str) -> float:
    """
    Score a segment without doing contrast-splitting again.
    Returns a continuous score (pos - neg).
    """
    s = normalize_space(text.lower())

    # 0) Explicit neutral phrases override
    for p in NEU_PHRASES:
        if phrase_hit(p, s):
            return 0.0

    pos = 0.0
    neg = 0.0

    # 1) Phrase-level signals
    for p in POS_PHRASES:
        if phrase_hit(p, s):
            pos += 2.0
    for p in NEG_PHRASES:
        if phrase_hit(p, s):
            neg += 2.5

    toks = tokenize(s)

    # 2) Token-level with bounded negation + intensifiers
    negation_window = 0  # number of tokens for which negation is active

    for i, t in enumerate(toks):
        if t in NEGATION_WORDS:
            negation_window = 2
            continue

        negation_pending = negation_window > 0
        if negation_window > 0:
            negation_window -= 1

        # lookback for intensifiers in previous 1-2 tokens
        prev = toks[max(0, i - 2):i]
        mult = 1.0
        if any(x in INTENSIFIERS for x in prev):
            mult *= 1.3

        # Treat ok/okay/fine/fint as neutral tokens
        if t in {"ok", "okay", "fine", "fint"}:
            continue

        if t in POS_WORDS:
            if negation_pending:
                neg += 1.0 * mult  # "not good"
            else:
                pos += 1.0 * mult
            continue

        if t in NEG_WORDS:
            if negation_pending:
                pos += 0.5 * mult  # "not bad" -> weak positive
            else:
                neg += 1.0 * mult
            continue

    return pos - neg


def score_text(text: str) -> int:
    s = normalize_space(text.lower())

    # Contrast split: "good but fast" -> often mixed/neutral or second-part-weighted
    split = split_on_contrast(s)
    if split:
        a, b = split
        sa = score_segment(a)
        sb = score_segment(b)

        # If opposite signs and both non-trivial -> neutral
        if sa != 0 and sb != 0 and (sa > 0) != (sb > 0):
            return 0

        # Otherwise: weight the second part a bit more (common in evals)
        raw = 0.8 * sa + 1.2 * sb
    else:
        raw = score_segment(s)

    # Wider neutral zone to avoid over-predicting
    if -1.2 < raw < 1.2:
        return 0

    return quantize_label(raw)


@app.post("/v1/sentiment")
async def analyze_sentiment(payload: TextInput):
    return {"score": score_text(payload.text)}