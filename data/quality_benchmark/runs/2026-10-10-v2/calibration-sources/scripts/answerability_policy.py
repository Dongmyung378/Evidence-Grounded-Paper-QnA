"""Question-aspect relevance guard using only the requested paper evidence.

Repeated source-derived title or model names can dominate cross-encoder scores.
This guard scores the requested aspect after removing those topical anchors.
Its threshold must be fitted on declared development data, never at runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from retrieval_common import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "config/answerability_policy.json"
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
TITLE_WORDS = "paper|study|research|model|method|approach|framework|system"
KOREAN_TITLE_WORDS = "논문|연구|모델|방법|기법|프레임워크|시스템"
GENERIC_ENGLISH = {"the", "this", "that", "these", "those", "what", "which", "how",
                   "when", "where", "why", "who", "does", "did", "is", "are", "was",
                   "were", "do", "has", "have", "its", "paper", "study", "research"}

# A conservative bilingual vocabulary for concrete requested attributes. This
# checks presence only, never claims entailment. Unknown attributes remain the
# cross-encoder's responsibility. No paper names or evaluation cases appear here.
REQUESTED_ATTRIBUTE_VOCABULARY = (
    ("symptoms", r"\bsymptoms?\b|\bclinical signs?\b|증상|임상 징후",
     r"\bsymptoms?\b|\bclinical signs?\b|\bmanifestations?\b|\b(?:cough|fever|nausea|pain|fatigue|dyspnea|rash|hemorrhage)\b|증상|임상 징후|기침|발열|메스꺼움|통증|피로|호흡곤란|발진|출혈"),
    ("human_participants", r"\bhuman participants?\b|\bhuman subjects?\b|\bpatients?\b|인간 참여자|사람.{0,4}참여자|환자",
     r"\bparticipants?\b|\bsubjects?\b|\bpatients?\b|\bvolunteers?\b|참여자|피험자|환자"),
    ("funding", r"\bfunded\b|\bfunding\b|\bfunders?\b|\bsponsors?\b|자금|연구비|후원",
     r"\bfund(?:ed|ing|s|ers?)?\b|\bsponsor\w*\b|\bgrants?\b|\bfinanc\w*\b|\bsupport(?:ed)? (?:by|from)\b|자금|연구비|후원|지원 기관"),
    ("legal_safeguards", r"\blegal safeguards?\b|\blegal protections?\b|법적.{0,8}보호|법률.{0,8}보호",
     r"\blegal\b|\bregulat\w*\b|\bcomplian\w*\b|\bconsent\b|\bprivacy\b|\banonym\w*\b|법적|법률|규제|동의|개인정보|익명"),
    ("ethical_approval", r"\bethic\w* approv\w*\b|\binstitutional review\b|윤리.{0,8}승인|기관.{0,8}심의",
     r"\bethic\w*\b|\binstitutional review\b|\bIRB\b|윤리|기관.{0,8}심의"),
    ("memory", r"\bmemory\b|\bRAM\b|메모리",
     r"\bmemory\b|\bRAM\b|\bstorage\b|메모리|저장 공간"),
    ("latency", r"\blatency\b|\binference (?:time|speed)\b|추론.{0,4}(?:시간|속도)|지연 시간",
     r"\blatency\b|\binference\b|\bspeed\b|\bmilliseconds?\b|\bruntime\b|추론|속도|지연|실행 시간"),
    ("energy", r"\benergy\b|\bpower consumption\b|에너지|전력 소모",
     r"\benergy\b|\bpower\b|\bwatts?\b|에너지|전력"),
    ("hardware", r"\bhardware\b|\bprocessors?\b|하드웨어|프로세서",
     r"\bhardware\b|\bprocessors?\b|\bCPU\b|\bGPU\b|\bTPU\b|하드웨어|프로세서"),
    ("uncertainty", r"\buncertaint\w*\b|\bconfidence intervals?\b|불확실성|신뢰 구간",
     r"\buncertaint\w*\b|\bconfidence\b|\bvariance\b|\bcredible\b|불확실성|신뢰|분산"),
    ("computational_complexity", r"\bcomputational complexity\b|\btime complexity\b|계산 복잡도|시간 복잡도",
     r"\bcomplexity\b|\bcomputation\w*\b|\bO\s*\(|복잡도|계산"),
)


def load_answerability_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or type(config.get("enabled")) is not bool:
        raise ValueError("Unsupported answerability policy configuration")
    if config.get("query_transform") not in {
        "source_derived_topic_debias_v1", "source_derived_topic_debias_v2"
    }:
        raise ValueError("Unsupported answerability query transformation")
    if config.get("comparison") != "accept_if_greater_than_or_equal":
        raise ValueError("Unsupported answerability score comparison")
    if config["enabled"]:
        threshold = config.get("minimum_aspect_score")
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold):
            raise ValueError("An enabled answerability guard requires a finite calibrated threshold")
        calibration = config.get("calibration") or {}
        if calibration.get("split") != "dev" or calibration.get("execution_mode") != "real_models":
            raise ValueError("Answerability threshold must come from real-model dev-only calibration")
    return config


def _legacy_aspect_query(question, evidence):
    """Remove repeated in-source anchors without consulting benchmark labels.

    Requested attributes remain in the question. A generic whole-paper question
    with no remaining aspect is not rejected by this additional guard.
    """
    source = " ".join(item.get("text", "") for item in evidence)
    anchors = []
    for token in TOKEN.findall(question):
        if token.casefold() in GENERIC_ENGLISH or not any(char.isupper() for char in token):
            continue
        occurrences = re.findall(rf"\b{re.escape(token)}\b", source, flags=re.I)
        if len(occurrences) >= 2 and token.casefold() not in {item.casefold() for item in anchors}:
            anchors.append(token)
    # Korean identifiers can also be recognized from repeated source title phrases.
    for match in re.finditer(rf"([가-힣A-Za-z0-9_-]+)\s+(?:{KOREAN_TITLE_WORDS})", question):
        token = match.group(1)
        if token in {"어떤", "해당", "본", "이", "제안된", "기존", "모든"}:
            continue
        if len(re.findall(rf"(?<!\w){re.escape(token)}(?!\w)", source, flags=re.I)) >= 2:
            if token not in anchors:
                anchors.append(token)
    transformed = question
    removed_phrases = []
    for anchor in sorted(anchors, key=len, reverse=True):
        escaped = re.escape(anchor)
        # Descriptors are removed only as part of a source-anchored title phrase.
        english = re.compile(rf"\b{escaped}(?:\s+[A-Za-z][A-Za-z0-9_-]*){{0,2}}\s+(?:{TITLE_WORDS})\b(?:'s)?", re.I)
        korean = re.compile(rf"(?<!\w){escaped}(?:\s+[가-힣A-Za-z0-9_-]+){{0,2}}\s+(?:{KOREAN_TITLE_WORDS})(?:은|는|이|가|의|에|을|를)?")
        for pattern in (english, korean):
            removed_phrases.extend(match.group() for match in pattern.finditer(transformed))
            transformed = pattern.sub(" ", transformed)
        transformed = re.sub(rf"\b{escaped}\b(?:'s)?", " ", transformed, flags=re.I)
    transformed = re.sub(r"\b(?:the|this|that|proposed|present|current|paper|study|research)\b", " ", transformed, flags=re.I)
    transformed = re.sub(r"(?<![가-힣])(?:이|본|해당|제안된)\s+", " ", transformed)
    transformed = re.sub(r"(?:논문|연구)(?:은|는|이|가|의|에|을|를)?", " ", transformed)
    transformed = " ".join(transformed.split()).strip(" ?.,:;")
    meaningful = [token for token in TOKEN.findall(transformed) if token.casefold() not in GENERIC_ENGLISH]
    has_aspect = bool(meaningful or re.search(r"[가-힣]{2,}", transformed))
    return {"query": transformed, "removed_anchors": anchors,
            "removed_title_phrases": removed_phrases, "has_specific_aspect": has_aspect}


def _topic_anchors(question, evidence):
    """Identify repeated document names while preserving scientific acronyms."""
    source = " ".join(item.get("text", "") for item in evidence)
    anchors = []
    for token in TOKEN.findall(question):
        if token.casefold() in GENERIC_ENGLISH or not any(char.isupper() for char in token):
            continue
        # Scientific acronyms are often the requested construct itself.
        if len(token) > 1 and token.isupper():
            continue
        occurrences = re.findall(rf"\b{re.escape(token)}\b", source, flags=re.I)
        title_context = re.search(
            rf"\b{re.escape(token)}(?:\s+[A-Za-z][A-Za-z0-9_-]*){{0,2}}\s+(?:{TITLE_WORDS})\b",
            question, re.I,
        ) or re.search(
            rf"(?<!\w){re.escape(token)}(?:\s+[가-힣A-Za-z0-9_-]+){{0,2}}\s+(?:{KOREAN_TITLE_WORDS})",
            question, re.I,
        ) or re.search(
            rf"\b{re.escape(token)}(?:\s+[A-Za-z][A-Za-z0-9_-]*){{0,2}}\s+(?:{TITLE_WORDS})\b",
            source, re.I,
        )
        if len(occurrences) >= 2 and title_context and token.casefold() not in {item.casefold() for item in anchors}:
            anchors.append(token)
    for match in re.finditer(rf"([가-힣A-Za-z0-9_-]+)\s+(?:{KOREAN_TITLE_WORDS})", question):
        token = match.group(1)
        if token in {"어떤", "해당", "본", "이", "제안된", "기존", "모든"} or token.isupper():
            continue
        if len(re.findall(rf"(?<!\w){re.escape(token)}(?!\w)", source, re.I)) >= 2 and token not in anchors:
            anchors.append(token)
    return anchors


def aspect_query(question, evidence):
    """Preserve a grammatical question and isolate its requested aspect.

    Both views are scored, so a cause, instrument or repeated title cannot
    replace evidence about the requested result. No labels enter this transform.
    """
    anchors = _topic_anchors(question, evidence)
    natural = question.strip()
    focus = question.strip()
    removed_phrases = []

    def generic_title(match):
        suffix = (match.group(2) or "") if len(match.groups()) == 2 else ""
        return match.group(1) + suffix

    for anchor in sorted(anchors, key=len, reverse=True):
        escaped = re.escape(anchor)
        english = re.compile(rf"\b{escaped}(?:\s+[A-Za-z][A-Za-z0-9_-]*){{0,2}}\s+({TITLE_WORDS})\b", re.I)
        korean = re.compile(rf"(?<!\w){escaped}(?:\s+[가-힣A-Za-z0-9_-]+){{0,2}}\s+({KOREAN_TITLE_WORDS})(은|는|이|가|의|에|을|를|과|와)?")
        for pattern in (english, korean):
            removed_phrases.extend(match.group() for match in pattern.finditer(focus))
            natural = pattern.sub(generic_title, natural)
            focus = pattern.sub(" ", focus)
        natural = re.sub(rf"(?<!\w){escaped}(?!\w)(?:'s)?", " ", natural, flags=re.I)
        focus = re.sub(rf"(?<!\w){escaped}(?!\w)(?:'s)?", " ", focus, flags=re.I)
    # Retain grammar in both languages rather than stripping subject nouns and
    # leaving "What method does use" or an orphaned Korean case particle.
    focus = " ".join(natural.split()).strip(" ?.,:;")
    if re.match(r"^(?:what|which|how many)\b", focus, re.I):
        focus = re.split(r"\s+(?:by|using|via)\s+", focus, maxsplit=1, flags=re.I)[0].strip()
    # Korean instrumental clauses can precede the requested subject.
    instrumental = re.search(r"\S*(?:으로|로)\s+(.+)", focus)
    if instrumental and re.search(r"(?:무엇|어떤|어디|얼마)", instrumental.group(1)):
        focus = instrumental.group(1)
    focus = re.sub(r"\s+(?:by|in|of|from|to)$", "", focus, flags=re.I)
    natural = " ".join(natural.split())
    meaningful = [token for token in TOKEN.findall(focus) if token.casefold() not in GENERIC_ENGLISH]
    return {"query": focus, "natural_query": natural, "removed_anchors": anchors,
            "removed_title_phrases": removed_phrases,
            "has_specific_aspect": bool(meaningful or re.search(r"[가-힣]{2,}", focus))}


def _scoring_texts(evidence, anchors):
    """Mask names in the scoring view while preserving original evidence."""
    texts = []
    for item in evidence:
        text = item.get("text", "").strip()
        if not text:
            continue
        for anchor in anchors:
            text = re.sub(rf"(?<!\w){re.escape(anchor)}(?!\w)", "", text, flags=re.I)
        texts.append(" ".join(text.split()))
    return texts


def requested_attribute_coverage(question, evidence):
    """Reject a known concrete request only when its entire concept is absent.

    A source mention is necessary for these concepts, never sufficient for a
    correct answer. Negation and predicate support are validated downstream.
    """
    source = " ".join(item.get("text", "") for item in evidence)
    requested, missing = [], []
    for name, question_pattern, source_pattern in REQUESTED_ATTRIBUTE_VOCABULARY:
        if re.search(question_pattern, question, re.I):
            requested.append(name)
            if not re.search(source_pattern, source, re.I):
                missing.append(name)
    return {"requested_attributes": requested, "missing_requested_attributes": missing,
            "requested_attribute_coverage": not missing}


def score_answerability(question, evidence, selector_model, query_transform="source_derived_topic_debias_v2"):
    legacy = query_transform == "source_derived_topic_debias_v1"
    if query_transform not in {"source_derived_topic_debias_v1", "source_derived_topic_debias_v2"}:
        raise ValueError("Unsupported answerability query transformation")
    transformed = _legacy_aspect_query(question, evidence) if legacy else aspect_query(question, evidence)
    coverage = {} if legacy else requested_attribute_coverage(transformed["query"], evidence)
    texts = ([item.get("text", "").strip() for item in evidence if item.get("text", "").strip()]
             if legacy else _scoring_texts(evidence, transformed["removed_anchors"]))
    metadata = {"score_policy": "legacy_focused_only" if legacy else "minimum_of_full_and_focused_request",
                "source_anchors_masked": [] if legacy else transformed["removed_anchors"]}
    if not texts or not transformed["has_specific_aspect"]:
        return {**transformed, **metadata, **coverage, "max_aspect_score": None, "evidence_count": len(texts)}
    queries = list(dict.fromkeys([transformed["query"]] if legacy else
                                [transformed["query"], transformed["natural_query"]]))
    scores = selector_model.predict([(query, text) for query in queries for text in texts],
                                    batch_size=8, show_progress_bar=False, convert_to_numpy=True)
    values = scores.reshape(-1).tolist()
    if len(values) != len(texts) * len(queries) or any(not math.isfinite(float(value)) for value in values):
        raise RuntimeError("Answerability scorer returned missing or non-finite scores")
    maxima = [max(float(value) for value in values[start:start + len(texts)])
              for start in range(0, len(values), len(texts))]
    return {**transformed, **metadata, **coverage, "max_aspect_score": min(maxima),
            "focused_request_max_score": maxima[0], "full_question_max_score": maxima[-1],
            "evidence_count": len(texts)}


def evaluate_answerability(question, evidence, selector_model, config=None):
    config = config or load_answerability_config()
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not config["enabled"]:
        return {"abstain": False, "reason_code": "policy_disabled", "enabled": False,
                "config_fingerprint": fingerprint}
    signals = score_answerability(question, evidence, selector_model, config["query_transform"])
    threshold = float(config["minimum_aspect_score"])
    if not signals["evidence_count"]:
        reason = "no_aspect_evidence"
    elif signals.get("missing_requested_attributes"):
        reason = "requested_attribute_absent"
    elif not signals["has_specific_aspect"]:
        reason = None
    elif signals["max_aspect_score"] < threshold:
        reason = "requested_aspect_not_supported"
    else:
        reason = None
    return {"abstain": reason is not None, "reason_code": reason, "enabled": True,
            "signals": signals, "minimum_aspect_score": threshold,
            "config_fingerprint": fingerprint}
