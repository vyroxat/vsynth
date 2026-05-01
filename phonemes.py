"""
phonemes.py — English text → ARPABET phoneme string
Uses NLTK CMU Pronouncing Dictionary with a lightweight G2P fallback.
Numbers are converted to spoken words before lookup.
"""
import re
import nltk

# ── Number → words ─────────────────────────────────────────────────────────────
_ONES = [
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
    "seventeen","eighteen","nineteen",
]
_TENS = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

def _int_to_words(n: int) -> str:
    if n < 0:
        return "negative " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        rest = "" if n % 10 == 0 else " " + _ONES[n % 10]
        return _TENS[n // 10] + rest
    if n < 1000:
        rest = "" if n % 100 == 0 else " " + _int_to_words(n % 100)
        return _ONES[n // 100] + " hundred" + rest
    if n < 1_000_000:
        rest = "" if n % 1000 == 0 else " " + _int_to_words(n % 1000)
        return _int_to_words(n // 1000) + " thousand" + rest
    if n < 1_000_000_000:
        rest = "" if n % 1_000_000 == 0 else " " + _int_to_words(n % 1_000_000)
        return _int_to_words(n // 1_000_000) + " million" + rest
    return str(n)  # fallback for huge numbers

def _number_token_to_words(tok: str) -> str:
    """Convert a numeric token like '6', '3.14', '-7', '1,000' to spoken words."""
    tok = tok.replace(",", "")  # strip thousands separators
    try:
        if "." in tok:
            integer_part, decimal_part = tok.split(".", 1)
            words = _int_to_words(int(integer_part)) + " point"
            for digit in decimal_part:
                words += " " + _ONES[int(digit)]
            return words
        else:
            return _int_to_words(int(tok))
    except (ValueError, IndexError):
        return tok  # not a plain number, return as-is

def _expand_numbers(text: str) -> str:
    """Replace all numeric tokens in text with their spoken-word equivalents."""
    # Match negative numbers, decimals, and integers (with optional thousands commas)
    return re.sub(
        r"-?\d[\d,]*(?:\.\d+)?",
        lambda m: _number_token_to_words(m.group()),
        text,
    )

# Download cmudict silently on first run
nltk.download("cmudict", quiet=True)
from nltk.corpus import cmudict

_CMU = cmudict.dict()

# Strip stress digits: AH0 -> AH, EH1 -> EH
def _strip_stress(phones):
    return [re.sub(r"\d+$", "", p) for p in phones]


# ── Simple rule-based G2P fallback ──────────────────────────────────────────
# Covers common English letter patterns for words not in cmudict.
_RULES = [
    # Digraphs first
    (r"ph",  "F"),
    (r"ck",  "K"),
    (r"ch",  "CH"),
    (r"sh",  "SH"),
    (r"th",  "DH"),
    (r"wh",  "W"),
    (r"ng",  "NG"),
    (r"qu",  "KW"),
    (r"gh",  ""),        # silent in night/though
    # Vowels
    (r"ou",  "AW"),
    (r"oo",  "UW"),
    (r"ee",  "IY"),
    (r"ea",  "IY"),
    (r"ai",  "EY"),
    (r"ay",  "EY"),
    (r"oa",  "OW"),
    (r"ow",  "OW"),
    (r"oi",  "OY"),
    (r"oy",  "OY"),
    (r"aw",  "AO"),
    (r"ew",  "UW"),
    (r"ue",  "UW"),
    (r"ie",  "IY"),
    # Silent e at end makes vowel long  (simplified)
    (r"a(?=\w*e\b)", "EY"),
    (r"i(?=\w*e\b)", "AY"),
    (r"o(?=\w*e\b)", "OW"),
    (r"u(?=\w*e\b)", "UW"),
    # Single letters
    (r"a",   "AE"),
    (r"e",   "EH"),
    (r"i",   "IH"),
    (r"o",   "AO"),
    (r"u",   "AH"),
    (r"y",   "IY"),
    (r"b",   "B"),
    (r"c",   "K"),
    (r"d",   "D"),
    (r"f",   "F"),
    (r"g",   "G"),
    (r"h",   "HH"),
    (r"j",   "JH"),
    (r"k",   "K"),
    (r"l",   "L"),
    (r"m",   "M"),
    (r"n",   "N"),
    (r"p",   "P"),
    (r"q",   "K"),
    (r"r",   "R"),
    (r"s",   "S"),
    (r"t",   "T"),
    (r"v",   "V"),
    (r"w",   "W"),
    (r"x",   "K S"),
    (r"z",   "Z"),
]
_RULES_COMPILED = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in _RULES]


def _g2p_fallback(word: str) -> list[str]:
    """Very rough letter-to-phoneme rules for OOV words."""
    s = word.lower()
    result = ""
    i = 0
    while i < len(s):
        matched = False
        for pattern, rep in _RULES_COMPILED:
            m = pattern.match(s, i)
            if m:
                if rep:
                    result += " " + rep if result else rep
                i = m.end()
                matched = True
                break
        if not matched:
            i += 1
    return result.split() if result.strip() else ["AH"]


def word_to_arpabet(word: str) -> list[str]:
    """Convert a single word to ARPABET phones, cmudict preferred."""
    key = word.lower().strip(".,!?;:'\"()-")
    if not key:
        return []
    if key in _CMU:
        return _strip_stress(_CMU[key][0])
    return _g2p_fallback(key)


def text_to_arpabet(text: str) -> str:
    """
    Convert a full English sentence to a klattsch ARPABET string.
    Punctuation pauses are mapped to klattsch , / . directives.
    Returns e.g. 'HH AH L OW , W ER L D'
    """
    # 1. Replace hyphens between word-chars FIRST so "6-7" → "6 7" (not "negative 7")
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    # 2. Now expand numbers: "6 7" → "six seven", "3.14" → "three point one four"
    text = _expand_numbers(text)
    # 3. Tokenise: keep words and sentence-break punctuation
    tokens = re.findall(r"[a-zA-Z']+|[,;.]", text)
    parts = []
    for tok in tokens:
        if tok in (",", ";"):
            parts.append(",")
        elif tok == ".":
            parts.append(".")
        else:
            phones = word_to_arpabet(tok)
            if phones:
                parts.extend(phones)
    return " ".join(parts)
