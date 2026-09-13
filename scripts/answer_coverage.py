"""질문별 정보 범위와 원문 연결을 고려한 추출형 답변 후보."""

from __future__ import annotations

import math
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter
from copy import deepcopy

from evidence_selector import evidence_object
from grounded_answer_contract import (
    AnswerValidationError, canonical_abstention_payload, detect_question_language,
    validate_answer_payload,
)
from grounded_generation import (
    _clean_extractive_sentence, _normalize_text, _usable_sentence, _intent_bonus,
    load_answer_generation_config, validate_plain_answer,
)
from retrieval_common import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "config/answer_coverage.json"


def load_coverage_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("seed") != 378:
        raise ValueError("Unsupported answer coverage configuration")
    for key in ("candidate_limit", "maximum_context_chunks", "maximum_sentences", "maximum_source_characters"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Invalid coverage limit: {key}")
    if not 0 < config["minimum_dense_similarity"] < 1:
        raise ValueError("Invalid dense similarity threshold")
    if config["maximum_context_chunks"] > 90 or config["maximum_sentences"] > 4:
        raise ValueError("Coverage budgets exceed the citation/answer contract")
    return config


def coverage_decision(result, decision, chunks, config):
    overview = overview_sources(result, chunks) if config["general_problem_overview"] else []
    if overview and select_problem_overview(overview):
        return {**decision, "abstain": False, "reason_code": None, "coverage_route": "abstract_overview"}, overview
    dense = result.get("retrieval_diagnostics", {}).get("top_dense_similarity")
    if not decision["abstain"] and (dense is None or not math.isfinite(float(dense)) or dense < config["minimum_dense_similarity"]):
        return {**decision, "abstain": True, "reason_code": "low_relevance", "coverage_route": "dense_support_guard"}, []
    return {**decision, "coverage_route": "retrieved_context"}, []


STOPWORDS = set("a an and are as at be been by can do does for from has have how in into is it its of on or our paper proposed that the their these this to using was we were what which who why with work works main".split())
FACETS = {
    "data": ("dataset", "data set", "benchmark", "cohort", "test set", "sample", "pipeline", "task schedule"),
    "comparison": ("baseline", "compared", "against", "versus", "whereas", "while", "outperform", "prior methods", "state-of-the-art"),
    "measurement": ("metric", "accuracy", "success", "efficiency", "diversity", "recovery", "utility", "throughput", "speedup", "memory", "performance"),
    "procedure": ("first", "second", "then", "stage", "step", "transfer", "extrapolat", "adapt", "construct"),
    "constraint": ("assum", "require", "condition", "constraint", "bound", "model", "construct"),
    "motivation": ("challenge", "problem", "lack", "limitation", "cannot", "non-identif", "address", "guarantee"),
}


def tokens(text):
    return {word.casefold() for word in re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
            if word.casefold() not in STOPWORDS and len(word) > 2}


def question_facets(question):
    lower = question.casefold()
    if any(term in lower for term in ("evaluat", "experiment", "what evidence", "평가", "실험", "어떤 근거")):
        return {"data", "comparison", "measurement"}
    if any(term in lower for term in ("trade-off", "tradeoff", "비교", "차이")):
        return {"comparison", "measurement"}
    if any(term in lower for term in ("assum", "가정", "required")):
        return {"constraint", "procedure"}
    if any(term in lower for term in ("how", "어떻게", "방법", "framework")):
        return {"procedure"}
    if any(term in lower for term in ("problem", "문제", "목표")):
        return {"motivation"}
    return set()


def is_general_problem_question(question):
    """외부 주어나 조건이 없는 논문 자체의 주요 문제 질문만 개요로 분류한다."""
    if detect_question_language(question) == "ko":
        return bool(re.fullmatch(
            r"(?:이|본)?\s*(?:논문|연구)(?:에서|이|의|가|는|은)?\s*"
            r"(?:(?:다루는|해결하려는|해결하는|다루고 있는)\s*)?(?:주요|핵심|중심)?\s*"
            r"문제(?:는|가)?\s*무엇(?:인가|입니까|인지)(?:요)?[?.\s]*", question.strip()))
    words = set(re.findall(r"[a-z]+", question.casefold()))
    allowed = set("what is are the main central key problem problems challenge challenges addressed tackled studied discussed in by this paper study research that deals with does address".split())
    return bool(words & {"problem", "problems", "challenge", "challenges"}) and bool(words & {"paper", "study", "research"}) and words <= allowed


def overview_sources(result, chunks):
    if not is_general_problem_question(result["query"]):
        return []
    abstract = sorted((c for c in chunks if c["paper_id"] == result["paper_id"]
                       and c["page"] <= 2 and "abstract" in (c.get("section") or "").casefold()),
                      key=lambda c: c["chunk_id"])
    return [evidence_object({"chunk": c, "rank": i, "score": 0.0}, i)
            for i,c in enumerate(abstract[:8], 1)]


def select_problem_overview(evidence):
    records = sentence_candidates(evidence)
    if not records:
        return []
    for row in records:
        row["cross_encoder_score"] = float(sum(k in row["text"].casefold()
            for k in ("problem", "challenge", "lack", "cannot", "limitation", "guarantee", "constraint", "instead", "however")))
    ranked = sorted(records, key=lambda r: (-r["cross_encoder_score"], r["page"], r["position"]))
    selected, ids, length = [], set(), 0
    for row in ranked:
        new_ids = {s["evidence_id"] for s in row["source_spans"]}
        if len(ids | new_ids) > 5 or length+len(row["text"]) > 1450:
            continue
        selected.append(row)
        ids |= new_ids
        length += len(row["text"])
        if len(selected) == 3:
            break
    return sorted(selected,key=lambda r:(r["page"],r["position"]))


def expand_answer_evidence(result, chunks, candidate_limit=12, max_chunks=30):
    """선택된 논문 안에서 검색 후보와 같은 페이지의 바로 옆 청크만 확장한다."""
    paper_id = result["paper_id"]
    paper = sorted((c for c in chunks if c["paper_id"] == paper_id),
                   key=lambda c: (c["page"], c["chunk_id"]))
    positions = {c["chunk_id"]: i for i, c in enumerate(paper)}
    selected = list(result["evidence"])
    seen = {e["chunk_id"] for e in selected}
    scores = {c["chunk_id"]: c for c in result["candidates"]}
    seeds = list(dict.fromkeys([e["chunk_id"] for e in selected] +
                              [c["chunk_id"] for c in result["candidates"][:candidate_limit]]))
    for chunk_id in seeds:
        if chunk_id not in positions:
            continue
        index = positions[chunk_id]
        for offset in (0, -1, 1):
            neighbor = index + offset
            if not 0 <= neighbor < len(paper):
                continue
            chunk = paper[neighbor]
            if chunk["page"] != paper[index]["page"] or chunk["chunk_id"] in seen:
                continue
            if len(selected) >= max_chunks:
                break
            source = scores.get(chunk["chunk_id"], scores.get(chunk_id, {}))
            rank = len(selected) + 1
            item = evidence_object({"chunk": chunk, "rank": rank,
                "score": float(source.get("reranker_score", 0)),
                "reranker_score": float(source.get("reranker_score", 0)),
                "retrieval_score": float(source.get("hybrid_score", 0))}, rank)
            item["context_origin"] = "retrieved_candidate" if offset == 0 else "same_page_neighbor"
            selected.append(item)
            seen.add(chunk["chunk_id"])
    return selected


def _sentence_spans(text):
    # 약어 내부 마침표를 문장 끝으로 처리하지 않는다.
    protected = text
    for match in reversed(list(re.finditer(r"\b(?:e\.g\.|i\.e\.|et al\.|Fig\.|Eq\.|Dr\.)", text))):
        protected = protected[:match.start()] + match.group().replace(".", "\u2024") + protected[match.end():]
    start = 0
    for match in re.finditer(r"[.!?](?:[\]\)])?(?:\s+|$)", protected):
        end = match.end()
        if end < len(text) and not re.match(r"[A-Z0-9(\[]", text[end:]):
            continue
        left = start + len(text[start:end]) - len(text[start:end].lstrip())
        right = end - len(text[start:end]) + len(text[start:end].rstrip())
        if right > left:
            yield left, right
        start = end


def sentence_candidates(evidence):
    """겹친 청크를 잇고 각 문장의 원문 청크와 문자 위치를 보존한다."""
    ordered = sorted(evidence, key=lambda e: (e.get("paper_id", ""), e["page"],
                     e.get("chunk_id", e["evidence_id"])))
    windows = []
    for item in ordered:
        text = _normalize_text(item["text"])
        overlap = 0
        if windows and windows[-1]["page"] == item["page"] and windows[-1]["section"] == item.get("section"):
            previous = windows[-1]["text"]
            for length in range(min(len(previous), len(text)), 23, -1):
                if previous.endswith(text[:length]):
                    overlap = length
                    break
        if overlap:
            window = windows[-1]
            start = len(window["text"]) - overlap
            window["text"] += text[overlap:]
            window["sources"].append((item, start, start + len(text)))
        else:
            windows.append({"text": text, "page": item["page"], "section": item.get("section"),
                            "sources": [(item, 0, len(text))]})
    records, seen = [], set()
    for window in windows:
        for start, end in _sentence_spans(window["text"]):
            sentence = window["text"][start:end]
            if not _usable_sentence(_clean_extractive_sentence(sentence), window["section"]) or sentence.casefold() in seen:
                continue
            if re.match(r"^(?:Figure|Fig\.|Table)\s*\d", sentence, re.I):
                continue
            if re.search(r"corresponding author|e-mail:|supported by|funded by|copyright|all rights reserved", sentence, re.I):
                continue
            if sentence.count("(") != sentence.count(")") or sentence.count("[") != sentence.count("]"):
                continue
            if len(sentence) > 900:
                continue
            spans = []
            covered = start
            for item, a, b in window["sources"]:
                if a <= covered < b:
                    stop = min(end, b)
                    spans.append({"evidence_id": item["evidence_id"], "start": covered-a, "end": stop-a})
                    covered = stop
                if covered == end:
                    break
            if covered != end:
                continue
            seen.add(sentence.casefold())
            records.append({"sentence_id": f"s-{len(records)+1:03d}", "text": sentence,
                            "evidence_id": spans[0]["evidence_id"], "source_spans": spans,
                            "page": window["page"], "section": window["section"],
                            "position": start, "evidence_rank": min(i["rank"] for i, a, b in window["sources"] if a < end and b > start)})
    return records


def rank_answer_sentences(question, evidence, model):
    records = sentence_candidates(evidence)
    if not records:
        return []
    contexts = []
    for index, row in enumerate(records):
        nearby = [r["text"] for r in records[max(0,index-1):index+2]
                  if r["page"] == row["page"] and r["section"] == row["section"]]
        contexts.append(" ".join(nearby)[:1800])
    pairs = [(question, r["text"]) for r in records] + [(question, c) for c in contexts]
    scores = model.predict(pairs, batch_size=16,
                           show_progress_bar=False, convert_to_numpy=True).reshape(-1).tolist()
    if len(scores) != 2*len(records):
        raise RuntimeError("Sentence reranker output count mismatch")
    for i, row in enumerate(records):
        row["sentence_score"] = float(scores[i])
        row["context_score"] = float(scores[i+len(records)])
        row["cross_encoder_score"] = .35*row["sentence_score"] + .65*row["context_score"]
    return [r for r in records if math.isfinite(r["cross_encoder_score"])]


def select_answer_sentences(question, records, max_sentences=4, max_chars=1450):
    """질문 관련도, 새로운 정보, 중복 억제로 제한된 답변 범위를 구성한다."""
    if not records:
        return []
    query_tokens = tokens(question)
    facets = question_facets(question)
    if any(term in question.casefold() for term in ("focus", "중점")):
        max_sentences = min(max_sentences, 2)
    if "constraint" in facets:
        max_sentences = min(max_sentences, 3)
    df = Counter(t for r in records for t in tokens(r["text"]))
    selected, covered, used_ids, used_tokens = [], set(), set(), set()
    total = 0
    remaining = list(records)
    top_score = max(r["cross_encoder_score"] for r in records)
    while remaining and len(selected) < max_sentences:
        options = []
        for r in remaining:
            ts = tokens(r["text"])
            ids = {s["evidence_id"] for s in r["source_spans"]}
            if len(used_ids | ids) > 5 or total + len(r["text"]) > max_chars:
                continue
            redundancy = max((len(ts & tokens(p["text"])) / max(1, min(len(ts), len(tokens(p["text"])))) for p in selected), default=0)
            if redundancy > .60:
                continue
            present = {f for f in facets if any(k in r["text"].casefold() for k in FACETS[f])}
            lexical = sum(math.log(1 + len(records)/(1+df[t])) for t in query_tokens & ts)
            new_terms = sum(math.log(1 + len(records)/(1+df[t])) for t in query_tokens & (ts-used_tokens))
            abstract_bonus = 2.0 if "abstract" in (r["section"] or "").casefold() else 0.0
            bonus = _intent_bonus(question, r)
            if "data" in facets:
                bonus += 3*len(present)
                if not re.search(r"evaluat|experiment|empirically|compar|measur|report|test|benchmark|achiev", r["text"], re.I):
                    bonus -= 8
            if facets == {"procedure"} and re.match(r"(?:Firstly|Secondly|First|Second|Then)[,: ]", r["text"]):
                bonus += 6
            score = r["cross_encoder_score"] + bonus + .20 * lexical + 2.0*len(present-covered) + .2*new_terms - 10*redundancy + abstract_bonus
            options.append((score, -r["evidence_rank"], -r["position"], r, ts, ids, present))
        if not options:
            break
        best = max(options, key=lambda row: row[:3])
        r, ts, ids, present = best[3:]
        selected.append({**r, "coverage_score": best[0]})
        remaining.remove(r)
        covered |= present
        used_tokens |= ts
        used_ids |= ids
        total += len(r["text"])
        if facets == {"data", "comparison", "measurement"} and facets <= covered and len(selected) >= 2:
            break
    return selected


def generate_coverage_answer(translator, selector, question, evidence, config=None, records=None, selected=None, coverage_config=None):
    language = detect_question_language(question)
    config = deepcopy(config or load_answer_generation_config())
    config["answer"]["maximum_sentences"] = 5
    config["answer"]["maximum_characters"] = 2000
    coverage_config = coverage_config or load_coverage_config()
    if selected is None:
        records = rank_answer_sentences(question, evidence, selector) if records is None else records
        selected = select_answer_sentences(question, records,
                    max_sentences=coverage_config["maximum_sentences"],
                    max_chars=coverage_config["maximum_source_characters"])
    ids = list(dict.fromkeys(s["evidence_id"] for r in selected for s in r["source_spans"]))
    cited = [e for e in evidence if e["evidence_id"] in ids]
    attempts = []
    try:
        if not selected:
            raise AnswerValidationError("No complete source sentence supports this answer")
        sources = {e["evidence_id"]: _normalize_text(e["text"]) for e in evidence}
        for row in selected:
            parts = []
            for span in row["source_spans"]:
                source = sources.get(span["evidence_id"], "")
                if not 0 <= span["start"] < span["end"] <= len(source):
                    raise AnswerValidationError("Source span is outside its original chunk")
                parts.append(source[span["start"]:span["end"]])
            if "".join(parts) != row["text"]:
                raise AnswerValidationError("Selected text differs from the original source spans")
        sentences = [_clean_extractive_sentence(r["text"]) for r in selected]
        translated = translator.translate(sentences) if language == "ko" else sentences
        if len(translated) != len(selected):
            raise AnswerValidationError("Translation count does not match the source sentences")
        accepted, accepted_text = [], []
        for source, text in zip(selected, translated):
            try:
                if language == "ko" and any(unicodedata.category(c).startswith("L") and
                        not any(name in unicodedata.name(c, "") for name in ("LATIN", "HANGUL", "GREEK"))
                        for c in text):
                    raise AnswerValidationError("Translation contains an unexpected writing system")
                validate_plain_answer(text, question, [source], language, config)
            except AnswerValidationError as error:
                attempts.append({"status": "rejected_sentence", "sentence_id": source["sentence_id"], "error": str(error)})
                continue
            accepted.append(source)
            accepted_text.append(text)
        selected = accepted
        if not selected:
            raise AnswerValidationError("Every source sentence failed answer validation")
        ids = list(dict.fromkeys(s["evidence_id"] for r in selected for s in r["source_spans"]))
        cited = [e for e in evidence if e["evidence_id"] in ids]
        answer = " ".join(accepted_text)
        response = validate_plain_answer(answer, question, selected, language, config)
        response["evidence_ids"] = ids
        validate_answer_payload(response, cited, language)
        fallback = False
    except (AnswerValidationError, ValueError, RuntimeError) as error:
        response = canonical_abstention_payload(language, "선택한 원문과 답변의 일치 여부를 검증하지 못했습니다." if language == "ko" else "The answer could not be verified against the selected source text.")
        attempts.append({"status": "invalid", "error": str(error)})
        fallback = True
    return {"response": response, "selected_sentences": selected, "answer_evidence": cited,
            "attempts": attempts, "fallback_used": fallback}
