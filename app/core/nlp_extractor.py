"""
nlp_extractor.py
----------------
Extracts symptom keywords from free-text user input.

The symptom lexicon is now generated dynamically from the CSV dataset
at initialisation time instead of being hardcoded.

Pipeline:
  1. Read all symptom strings from symptom_disease.csv
  2. Auto-generate synonym variants for each canonical symptom
     (e.g. "burning urination" → ["burn urinating", "burning when peeing", ...])
  3. Apply longest-match extraction with negation detection
"""

import re
import csv
import os
from typing import NamedTuple
import spacy

try:
    from rapidfuzz import process as fuzz_process, fuzz
    _FUZZY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FUZZY_AVAILABLE = False

# Minimum similarity score (0-100) to accept a fuzzy match
FUZZY_THRESHOLD = 85

# Optional: use spaCy for better dependency parsing and tense detection.
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    spacy = None  # type: ignore
    _SPACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Synonym generator — turns dataset symptom names into lexicon entries
# ---------------------------------------------------------------------------

# Hand-authored synonym expansions for common medical terms.
# These supplements the auto-generated variants from the CSV column names.
_MANUAL_SYNONYMS: dict[str, list[str]] = {
    "headache":               ["headache", "head ache", "head pain", "head hurts", "head is pounding",
                               "head is throbbing", "head hurting", "my head hurts", "cephalgia"],
    "throbbing pain":         ["throbbing", "pulsating", "pounding pain", "pulsing pain"],
    "one-sided pain":         ["one side", "one-sided", "left side of head", "right side of head", "half my head"],
    "pressure around forehead":["pressure", "band around head", "tight head", "squeezing",
                                "pressure in head", "head feels tight"],
    "visual aura":            ["aura", "visual disturbance", "flashing lights", "zigzag lines",
                               "blurred vision before headache", "visual changes"],
    "dizziness":              ["dizzy", "dizziness", "lightheaded", "light-headed", "vertigo",
                               "spinning", "unsteady", "off balance"],
    "sensitivity to light":   ["sensitive to light", "light hurts eyes", "photophobia",
                               "light sensitivity", "bright light painful", "can't stand light"],
    "sensitivity to sound":   ["sensitive to sound", "noise hurts", "phonophobia",
                               "sound sensitivity", "loud noises bother me"],
    "runny nose":             ["runny nose", "runny", "nose is running", "nasal discharge",
                               "nose dripping", "dripping nose"],
    "sneezing":               ["sneezing", "sneeze", "sneezes", "keep sneezing"],
    "congestion":             ["congested", "stuffy nose", "blocked nose", "nasal congestion",
                               "can't breathe through nose", "stuffed up"],
    "sore throat":            ["sore throat", "throat pain", "throat hurts", "throat is sore",
                               "painful swallowing", "throat ache", "scratchy throat"],
    "difficulty swallowing":  ["hard to swallow", "difficulty swallowing", "painful swallowing",
                               "swallowing hurts", "can't swallow"],
    "cough":                  ["cough", "coughing", "dry cough", "wet cough", "chesty cough"],
    "shortness of breath":    ["short of breath", "breathless", "can't breathe",
                               "difficulty breathing", "out of breath", "hard to breathe", "breathing difficulty"],
    "chest pain":             ["chest pain", "chest hurts", "pain in chest", "chest ache", "chest discomfort"],
    "chest tightness":        ["chest tight", "tightness in chest", "chest feels tight", "chest pressure"],
    "heartburn":              ["heartburn", "heart burn", "burning in chest", "chest burning",
                               "acid in throat", "burning sensation chest"],
    "racing heart":           ["racing heart", "heart pounding", "palpitations", "heart beating fast",
                               "heart racing", "rapid heartbeat"],
    "nausea":                 ["nausea", "nauseous", "feel sick", "feeling sick", "want to vomit",
                               "queasy", "stomach feels sick", "feel nauseous"],
    "vomiting":               ["vomiting", "vomited", "threw up", "throwing up", "been sick",
                               "puking", "vomit"],
    "diarrhoea":              ["diarrhoea", "diarrhea", "loose stool", "watery stool", "loose stools",
                               "runny stool", "frequent stool"],
    "stomach cramps":         ["stomach cramps", "stomach pain", "abdominal pain", "belly pain",
                               "stomach ache", "tummy ache", "abdominal cramps", "gut pain"],
    "stomach pain":           ["stomach pain", "abdominal pain", "belly ache", "tummy pain"],
    "burning urination":      ["burning when urinating", "burning urination", "pain when peeing",
                               "stinging urine", "burning pee", "it burns when i pee",
                               "burns when i urinate", "pain urinating"],
    "frequent urination":     ["frequent urination", "urinating often", "need to urinate often",
                               "peeing a lot", "going to toilet often", "urgency to urinate",
                               "going bathroom often"],
    "body aches":             ["body aches", "muscle aches", "all over aches", "aching",
                               "sore all over", "body pain", "aching body"],
    "back pain":              ["back pain", "backache", "back ache", "lower back pain",
                               "my back hurts", "back is sore", "lumbar pain"],
    "joint pain":             ["joint pain", "joints hurt", "arthralgia", "achy joints"],
    "muscle pain":            ["muscle pain", "muscle soreness", "sore muscles", "muscle ache",
                               "muscles hurt", "myalgia"],
    "fever":                  ["fever", "temperature", "high temperature", "febrile", "feel hot",
                               "running a temperature", "38 degrees", "39 degrees", "feverish",
                               "high fever", "fever and chills"],
    "chills":                 ["chills", "shivering", "shakes", "feeling cold", "rigors", "can't get warm"],
    "fatigue":                ["tired", "fatigue", "exhausted", "no energy", "lethargy", "weak",
                               "feeling run down", "wiped out", "tiredness", "exhaustion"],
    "sweating":               ["sweating", "sweaty", "night sweats", "excessive sweating", "perspiring"],
    "loss of appetite":       ["no appetite", "not hungry", "lost appetite", "don't want to eat",
                               "can't eat", "reduced appetite"],
    "rash":                   ["rash", "skin rash", "red rash", "hives", "itchy rash", "spots on skin"],
    "itchy skin":             ["itchy skin", "skin itch", "skin itching", "itchiness", "pruritus"],
    "dry skin":               ["dry skin", "skin is dry", "flaky skin"],
    "red eyes":               ["red eyes", "pink eye", "bloodshot", "eyes are red", "eye redness"],
    "itchy eyes":             ["itchy eyes", "eye itch", "eyes itch", "itching eyes"],
    "watery eyes":            ["watery eyes", "eyes watering", "tearing", "tears"],
    "blurred vision":         ["blurred vision", "blurry vision", "vision blurred", "fuzzy vision"],
    "thirst":                 ["thirsty", "thirst", "very thirsty", "drinking lots", "increased thirst"],
    "frequent urination":     ["frequent urination", "urinating frequently", "peeing frequently"],
    "weight loss":            ["losing weight", "weight loss", "lost weight unintentionally", "unexplained weight loss"],
    "weight gain":            ["weight gain", "gaining weight", "putting on weight"],
    "trembling":              ["trembling", "shaking", "tremor", "hands shaking", "shaky"],
    "anxiety":                ["anxious", "anxiety", "feeling anxious", "nervousness", "on edge"],
    "depression":             ["depressed", "depression", "feeling depressed", "low mood", "hopeless"],
    "confusion":              ["confused", "confusion", "disoriented", "not thinking clearly"],
    "spinning sensation":     ["spinning sensation", "room spinning", "world spinning", "feel like spinning"],
    "yellow skin":            ["yellow skin", "yellowing skin", "jaundice", "skin turned yellow"],
    "yellow eyes":            ["yellow eyes", "eyes are yellow", "whites of eyes yellow"],
    "dark urine":             ["dark urine", "brown urine", "cola coloured urine", "dark yellow urine"],
    "pale skin":              ["pale skin", "pallor", "skin looks pale", "washed out"],
    "cold intolerance":       ["cold intolerance", "sensitive to cold", "always cold", "intolerant of cold"],
    "hair loss":              ["hair loss", "losing hair", "hair falling out", "alopecia", "baldness"],
    "constipation":           ["constipated", "constipation", "can't go to toilet", "hard stool"],
    "wheezing":               ["wheezing", "wheeze", "whistling breath", "chest wheezing"],
    "facial pain":            ["facial pain", "face pain", "face hurts", "facial discomfort"],
    "nasal congestion":       ["nasal congestion", "blocked nose", "stuffy nose", "congested nose"],
    "post-nasal drip":        ["post-nasal drip", "drip down throat", "mucus down throat"],
    "productive cough":       ["productive cough", "cough with phlegm", "cough bringing up mucus", "wet cough"],
    "high fever":             ["high fever", "very high temperature", "burning fever", "temperature of 39",
                               "temperature of 40"],
    "severe headache":        ["severe headache", "bad headache", "really bad headache", "excruciating headache"],
    "eye pain":               ["eye pain", "pain in eye", "painful eyes", "pain behind eyes"],
    "skin rash":              ["skin rash", "rash on skin", "red patches on skin", "skin eruption"],
}


def _auto_synonyms(canonical: str) -> list[str]:
    """
    Generate simple variants of a canonical symptom string.

    Includes basic normalization such as:
    - replacing underscores with spaces
    - handling simple plural forms

    Args:
        canonical (str): Base symptom string.

    Returns:
        list[str]: Generated synonym variants.
    """

    variants = {canonical}
    # Replace underscores if present
    variants.add(canonical.replace("_", " "))
    # Strip trailing 's'
    if canonical.endswith("s") and len(canonical) > 3:
        variants.add(canonical[:-1])
    return list(variants)


def build_lexicon_from_csv(csv_path: str) -> dict[str, list[str]]:
    """
    Build a symptom lexicon from a CSV dataset.

    Extracts symptom columns and maps each canonical symptom
    to a list of phrase variants, combining manual and auto-generated synonyms.

    Args:
        csv_path (str): Path to the dataset file.

    Returns:
        dict[str, list[str]]: Mapping of canonical symptoms to phrases.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    canonical_symptoms: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for i in range(1, 18):
                val = row.get(f"symptom_{i}", "").strip().lower()
                if val:
                    canonical_symptoms.add(val)

    lexicon: dict[str, list[str]] = {}

    # Start from manual synonyms (authoritative)
    for canonical, phrases in _MANUAL_SYNONYMS.items():
        lexicon[canonical] = list(dict.fromkeys(p.lower() for p in phrases))

    # Add CSV-derived symptoms not already in the manual list
    for sym in canonical_symptoms:
        if sym not in lexicon:
            lexicon[sym] = _auto_synonyms(sym)
        else:
            # Merge: ensure the canonical form itself is a phrase in the list
            if sym not in lexicon[sym]:
                lexicon[sym].insert(0, sym)

    return lexicon


# ---------------------------------------------------------------------------
# 2. Extractor class (same interface as before)
# ---------------------------------------------------------------------------

class ExtractionResult(NamedTuple):
    symptoms:    list   # canonical symptom names found (present/current)
    raw_mentions: list  # original phrases from user text
    negated:     list   # symptoms mentioned but negated ("no fever")
    tagged:      list   # list of dicts: {symptom, status: present|past|negated, raw, span}


class SymptomExtractor:
    def __init__(self, csv_path: str | None = None):
        """
        Initializes the SymptomExtractor.

        Builds the symptom lexicon either from a CSV file or fallback manual synonyms.
        Also prepares lookup structures and loads NLP model.

        Args:
            csv_path (str | None): Optional path to the dataset CSV file.

        Returns:
            None
        """
        # Build lexicon from CSV if path provided; fall back to manual synonyms
        if csv_path and os.path.exists(csv_path):
            lexicon = build_lexicon_from_csv(csv_path)
        else:
            lexicon = {k: list(dict.fromkeys(v)) for k, v in _MANUAL_SYNONYMS.items()}

        # Reverse lookup: phrase → canonical
        self.phrase_to_symptom: dict[str, str] = {}
        for canonical, phrases in lexicon.items():
            for phrase in phrases:
                self.phrase_to_symptom[phrase.lower()] = canonical

        # Sort by length descending (longer phrases matched first)
        self.sorted_phrases = sorted(
            self.phrase_to_symptom.keys(), key=len, reverse=True
        )

        # Pre-build a flat list of all phrases for fuzzy candidate lookup
        self._all_phrases: list[str] = list(self.phrase_to_symptom.keys())

        self.negation_patterns = re.compile(
            r"\b(no|not|without|don't have|doesn't have|haven't|hasn't|never|"
            r"no sign of|denies|absence of)\b",
            re.IGNORECASE
        )
        # Temporal cue patterns for simple heuristic detection
        self.past_cues = re.compile(r"\b(yesterday|last night|last week|last month|ago|previously|earlier|previous)\b", re.IGNORECASE)
        self.present_cues = re.compile(r"\b(now|currently|today|at the moment|right now|still|ongoing|since)\b", re.IGNORECASE)
        self.negation_tokens = {
            "no", "not", "without", "never", "none", "neither", "denies", "deny", "denied",
            "dont", "don't", "doesnt", "doesn't", "havent", "haven't", "hasnt", "hasn't",
            "didnt", "didn't", "cannot", "can't",
        }
        self.past_tokens = {
            "was", "were", "had", "previously", "earlier", "before", "yesterday",
            "ago", "last", "started", "initially", "former", "used to",
        }
        self.present_tokens = {
            "now", "currently", "today", "still", "ongoing", "present", "currently",
            "right now", "at the moment",
        }
        self.resolution_tokens = {"anymore", "resolved", "better", "improving", "improved", "fine"}
        self.clause_splitter = re.compile(r"\b(?:but|however|although|though|yet)\b|[.,;]", re.IGNORECASE)

        # Load spaCy model lazily if available; don't fail if not installed.
        self.nlp = None
        if _SPACY_AVAILABLE:
            try:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    self.nlp = spacy.blank("en")
            except Exception:
                self.nlp = None

        # Initialize spaCy for dependency-based negation parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[NLP] Model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
            self.nlp = None

        print(f"[NLP] Lexicon loaded: {len(lexicon)} canonical symptoms, "
              f"{len(self.phrase_to_symptom)} total phrases")

    def _is_negated(self, doc, start_char: int, end_char: int) -> bool:
        """
        Checks whether a detected symptom is negated in the sentence.

        Uses dependency parsing to determine if negation words are associated
        with the symptom.

        Args:
            doc (spacy.tokens.Doc): spaCy processed document.
            start_char (int): Start index of the symptom in text.
            end_char (int): End index of the symptom in text.

        Returns:
            bool: True if the symptom is negated, otherwise False.
        """
        if not doc:
            return False

        # Find tokens that overlap with the character range [start_char, end_char)
        symptom_tokens = [t for t in doc if t.idx >= start_char and t.idx < end_char]
        if not symptom_tokens:
            # Fallback for tokens that might start slightly before start_char (e.g. whitespace)
            symptom_tokens = [t for t in doc if t.idx + len(t.text) > start_char and t.idx < end_char]

        negation_words = {"no", "not", "without", "never", "deny", "denies", "absence", "negative"}

        for token in symptom_tokens:
            # 1. Direct children negation (e.g., "no fever")
            if any(child.dep_ == "neg" or child.lemma_.lower() in negation_words for child in token.children):
                return True
            
            # 2. Check ancestors (e.g., "denies pain" or "don't HAVE headache")
            curr = token
            while curr != curr.head:
                curr = curr.head
                # Check if the ancestor itself is a negation word (e.g. "DENIES pain")
                if curr.lemma_.lower() in negation_words:
                    return True
                # If the ancestor has a negation child (e.g. "don't HAVE headache")
                if any(child.dep_ == "neg" or child.lemma_.lower() in negation_words for child in curr.children):
                    return True
                if curr.pos_ == "VERB":
                    break

        return False

    def _fuzzy_match_token(self, token: str) -> tuple[str, str] | None:
        """
        Attempts to fuzzy match a token or phrase against known symptom phrases.

        Args:
            token (str): Input token or phrase.

        Returns:
            tuple[str, str] | None: Matched phrase and canonical symptom,
            or None if no match is found.
        """
        if not _FUZZY_AVAILABLE or len(token) < 3:
            return None
        result = fuzz_process.extractOne(
            token,
            self._all_phrases,
            scorer=fuzz.WRatio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if result is None:
            return None
        matched_phrase, score, _ = result
        return matched_phrase, self.phrase_to_symptom[matched_phrase]

    def extract(self, text: str) -> ExtractionResult:
        text_lower = text.lower()
        clause_ranges = self._build_clause_ranges(text_lower)

        # Use spaCy doc if available for better scope/tense detection
        doc = None
        if self.nlp:
            try:
                doc = self.nlp(text)
            except Exception:
                doc = None

        # Detect negation windows (fallback heuristic: 40 chars after negation keyword)
        negated_spans: set[tuple] = set()
        for match in self.negation_patterns.finditer(text_lower):
            start = match.end()
            end = min(start + 40, len(text_lower))
            negated_spans.add((start, end))

        def is_negated(pos: int, endpos: int) -> bool:
            # Prefer dependency-based negation detection when spaCy available
            if doc is not None:
                for token in doc:
                    if token.idx <= pos < token.idx + len(token.text):
                        # children with neg dependency
                        if any(ch.dep_ == "neg" for ch in token.children):
                            return True
                        # head may have negation
                        if any(ch.dep_ == "neg" for ch in token.head.children):
                            return True
                        # nearby lemmas
                        window_start = max(0, token.i - 3)
                        window_end = min(len(doc), token.i + 3)
                        for i in range(window_start, window_end):
                            if doc[i].lemma_.lower() in {"no", "not", "never", "without", "deny", "denies", "n't"}:
                                return True
                        return False # Trust spaCy if it's available and found the token
            # fallback: positional + clause-aware heuristic
            if any(s <= pos <= e for s, e in negated_spans):
                return True
            clause_text = self._clause_text_for_span(text_lower, pos, endpos, clause_ranges)
            return self._contains_negation(clause_text)

        def detect_status(pos: int, endpos: int) -> str:
            """Return one of: 'negated', 'past', 'present' (default)."""
            if is_negated(pos, endpos):
                return "negated"

            # Use spaCy verb tense if available
            if doc is not None:
                mention_token = None
                for token in doc:
                    if token.idx <= pos < token.idx + len(token.text):
                        mention_token = token
                        break

                if mention_token is not None:
                    verbs = []
                    head = mention_token.head
                    if head.pos_ == "VERB":
                        verbs.append(head)
                    for t in doc[max(0, mention_token.i - 4): mention_token.i + 4]:
                        if t.pos_ == "VERB":
                            verbs.append(t)

                    for v in verbs:
                        if v.tag_.startswith("VBD") or v.morph.get("Tense") == ["Past"]:
                            return "past"
                        if v.tag_.startswith("VBG") or v.morph.get("Tense") == ["Pres"]:
                            return "present"

            # Clause-aware fallback before local character window.
            clause_text = self._clause_text_for_span(text_lower, pos, endpos, clause_ranges)
            if self._contains_resolution(clause_text):
                return "past"
            if self._contains_past(clause_text) and not self._contains_present(clause_text):
                return "past"
            if self._contains_present(clause_text):
                return "present"
            # Fallback: look for temporal cue words near the mention in the raw text
            window_text = text_lower[max(0, pos - 60): min(len(text_lower), endpos + 60)]
            if self.past_cues.search(window_text):
                # "now/currently" in the same window should win when both exist.
                if not self.present_cues.search(window_text):
                    return "past"
            if self.present_cues.search(window_text):
                return "present"

            return "present"

        found_symptoms:   list[str] = []
        negated_symptoms: list[str] = []
        raw_mentions:     list[str] = []
        tagged:           list[dict] = []
        matched_positions: set[int] = set()

        for phrase in self.sorted_phrases:
            start = 0
            while True:
                idx = text_lower.find(phrase, start)
                if idx == -1:
                    break
                positions = set(range(idx, idx + len(phrase)))
                if positions & matched_positions:
                    start = idx + 1
                    continue

                canonical = self.phrase_to_symptom[phrase]
                raw = text[idx: idx + len(phrase)]
                raw_mentions.append(raw)
                matched_positions |= positions

                status = detect_status(idx, idx + len(phrase))

                if status == "negated":
                    if canonical not in negated_symptoms:
                        negated_symptoms.append(canonical)
                else:
                    if canonical not in found_symptoms:
                        found_symptoms.append(canonical)

                tagged.append({
                    "symptom": canonical,
                    "status": status,
                    "raw": raw,
                    "span": (idx, idx + len(phrase)),
                })

                start = idx + 1

        # ── Fuzzy fallback pass ──────────────────────────────────────────
        # Tokenize the text into word n-grams (1–3 words) that weren't
        # already covered by exact matching, then fuzzy-match each against
        # the full phrase lexicon.
        if _FUZZY_AVAILABLE:
            words = re.findall(r"[a-z']+", text_lower)
            # Generate trigrams → bigrams → unigrams (longest wins)
            candidates: list[tuple[int, str]] = []
            for n in (3, 2, 1):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i: i + n])
                    approx_pos = text_lower.find(ngram)
                    if approx_pos == -1:
                        continue
                    positions = set(range(approx_pos, approx_pos + len(ngram)))
                    if positions & matched_positions:
                        continue  # already matched exactly
                    candidates.append((approx_pos, ngram))

            seen_positions: set[int] = set()
            for approx_pos, ngram in candidates:
                positions = set(range(approx_pos, approx_pos + len(ngram)))
                if positions & seen_positions:
                    continue

                hit = self._fuzzy_match_token(ngram)
                if hit is None:
                    continue

                matched_phrase, canonical = hit
                seen_positions |= positions
                matched_positions |= positions
                raw_mentions.append(ngram)

                if self._is_negated(doc, approx_pos, approx_pos + len(ngram)):
                    if canonical not in negated_symptoms:
                        negated_symptoms.append(canonical)
                else:
                    if canonical not in found_symptoms:
                        found_symptoms.append(canonical)

        return ExtractionResult(
            symptoms     = found_symptoms,
            raw_mentions = raw_mentions,
            negated      = negated_symptoms,
            tagged       = tagged,
        )

    def _build_clause_ranges(self, text_lower: str) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = 0
        for match in self.clause_splitter.finditer(text_lower):
            end = match.start()
            if end > start:
                ranges.append((start, end))
            start = match.end()
        if start < len(text_lower):
            ranges.append((start, len(text_lower)))
        return ranges or [(0, len(text_lower))]

    def _clause_text_for_span(
        self,
        text_lower: str,
        start: int,
        end: int,
        clause_ranges: list[tuple[int, int]],
    ) -> str:
        for c_start, c_end in clause_ranges:
            if c_start <= start < c_end or c_start < end <= c_end:
                return text_lower[c_start:c_end]
        return text_lower[max(0, start - 60): min(len(text_lower), end + 60)]

    def _contains_negation(self, text_fragment: str) -> bool:
        words = set(re.findall(r"[a-z']+", text_fragment))
        return any(w in words for w in self.negation_tokens)

    def _contains_past(self, text_fragment: str) -> bool:
        if self.past_cues.search(text_fragment):
            return True
        words = set(re.findall(r"[a-z']+", text_fragment))
        return any(w in words for w in self.past_tokens)

    def _contains_present(self, text_fragment: str) -> bool:
        if self.present_cues.search(text_fragment):
            return True
        words = set(re.findall(r"[a-z']+", text_fragment))
        return any(w in words for w in self.present_tokens)

    def _contains_resolution(self, text_fragment: str) -> bool:
        words = set(re.findall(r"[a-z']+", text_fragment))
        return any(w in words for w in self.resolution_tokens)


# ---------------------------------------------------------------------------
# 3. Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    _here = pathlib.Path(__file__).parent.parent.parent
    csv_p = str(_here / "data" / "symptom_disease.csv")

    extractor = SymptomExtractor(csv_path=csv_p)
    
    print("\n" + "="*40)
    print("      SYMPTOM EXTRACTION TESTER")
    print("="*40)
    print("Type your symptoms (or 'quit' to exit)")

    while True:
        try:
            t = input("\n[Input]: ").strip()
            if not t or t.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break
                
            r = extractor.extract(t)
            print(f"  [Found]:   {r.symptoms}")
            if r.negated:
                print(f"  [Negated]: {r.negated}")
            if not r.symptoms and not r.negated:
                print("  [Result]:  No symptoms detected.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
