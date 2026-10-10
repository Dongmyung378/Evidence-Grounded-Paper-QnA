"""Select UI-ready evidence from reranked retrieval candidates."""

import hashlib
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[가-힣]+|[A-Za-z0-9]+")
PREDICTION_TASK_PATTERN = (
    r"\b(?:prediction|predictive|pretext) tasks?\b.*\b(?:aims?|identify|uses?|consists?)\b|"
    r"\bmaximiz\w* agreement\b"
)


def requested_source_facets(question):
    """Map explicit scientific aspects to source terms, without document names."""
    if not question:
        return []
    vocabulary = (
        ("projection", r"\bprojection\b|투영", r"\bprojection\b"),
        ("cropping", r"\bcrop\w*\b|자르기|크롭", r"\bcrop\w*\b"),
        ("flip", r"\bflip\w*\b|뒤집기|플립", r"\bflip\w*\b"),
        ("resize", r"\bresiz\w*\b|크기\s*조정|리사이즈", r"\bresiz\w*\b"),
        ("shortcut", r"\bshortcuts?\b|지름길|편법", r"\b(?:shortcuts?|exploit\w*|conjectur\w*)\b"),
        ("linear_classifier", r"\blinear classifier\w*\b|(?<!비)선형\s*분류", r"\blinear (?:classifier\w*|evaluation)\b"),
        ("scalar_parameter", r"\bscalar parameters?\b|스칼라\s*매개변수", r"\bscalar parameters?\b"),
        ("predicted_labels", r"\b(?:predicted labels?|class predictions?)\b|예측\s*(?:레이블|라벨|클래스)", r"\b(?:class predictions?|predicted labels?)\b"),
    )
    facets = [{"name": name, "source_pattern": source} for name, request, source in vocabulary
              if re.search(request, question, re.I)]
    if re.search(r"\b(?:prediction|predictive|pretext) tasks?\b|예측\s*과제", question, re.I):
        facets.append({"name": "prediction_task", "source_pattern": PREDICTION_TASK_PATTERN,
                       "plain_prose": True, "answer_context_only": True})
    if ("list" in question_kinds(question)
        and re.search(r"\b(?:augmentations?|transformations?)\b|증강|변환", question, re.I)):
        facets.append({"name": "augmentation_operations", "source_pattern":
                       r"(?=.*\b(?:augment\w*|transform\w*|crop\w*)\b)"
                       r"(?=.*\b(?:flip\w*|resiz\w*|blur\w*|distortion\w*)\b)"
                       r"(?=.*\b(?:appl\w*|sequential\w*|includes?|consists?)\b)"})
    return facets


def _facet_present(facet, text):
    if not facet.get("plain_prose"):
        return bool(re.search(facet["source_pattern"], text, re.I))
    return any(re.search(facet["source_pattern"], sentence, re.I)
               and not any(symbol in sentence for symbol in ("=", "∈", "⊆", "∆", "∂", "†"))
               and re.search(r"[.!?]\s*$", sentence)
               for sentence in re.split(r"(?<=[.!?])\s+", text))


def question_kinds(question):
    """Identify generic answer shapes without matching a particular paper."""
    if question is None:
        return set()
    if not isinstance(question, str):
        raise ValueError("question must be text or None")
    lowered = question.casefold()
    kinds = set()
    patterns = {
        "list": r"\b(?:list|enumerate|enumeration|what are)\b|"
        r"\b(?:which|what)(?:\s+(?:all|the|one|two|three|four|five|six|seven|eight|nine|ten|\d+))?"
        r"\s+(?:datasets|benchmarks|methods|components|inputs|steps|features|models|conditions|factors|augmentations|transformations)\b|"
        r"\b(?:which|what)\s+(?:default|standard)\s+(?:augmentations|transformations)\b|"
        r"목록|나열|열거|모두|각각|어떤\s*(?:데이터셋|벤치마크|요소|입력|단계|요인|증강|변환)|무엇들|"
        r"(?:[한두세네다섯여섯일곱여덟아홉열]+|\d+)\s*가지",
        "comparison": r"\b(?:compar\w*|versus|vs|differ\w*|trade[- ]?offs?)\b|비교|차이|달라|다르|다른",
        "numeric": r"\b(?:how much|how many|percentage|percent|numerical|number of|"
        r"speedup|accuracy|throughput)\b|얼마|몇|수치|정확도|처리량|속도|학습률|%",
        "limitations": r"\b(?:limitation\w*|restrict\w*|caveat\w*|failure\w*|"
        r"exceptions?|except|when does|under what conditions)\b|한계|제약|주의|실패|예외|적용\s*조건",
        "evaluation": r"\b(?:evaluat\w*|validat\w*)\b|"
        r"\b(?:what|which)\s+(?:\w+\s+){0,2}experiments?\b|"
        r"평가|검증|어떤\s*실험|무슨\s*실험|실험\s*(?:방법|방식|설계|절차)",
        "definition": r"\b(?:defin\w*|meaning|means?|mean by|refer\w* to)\b|"
        r"(?<![가-힣])정의|의\s*정의|의미|(?<![가-힣])뜻",
        "causal": r"\b(?:why|reasons?|causes?|causal|explain\w* why|shortcuts?)\b|왜|이유|원인|인과|지름길|편법",
    }
    for kind, pattern in patterns.items():
        if re.search(pattern, lowered):
            kinds.add(kind)
    return kinds


def fact_tokens(text):
    """Keep distinct quantities, technical identifiers and qualifications."""
    quantities = set(re.findall(r"[-+]?\d+(?:[.,]\d+)*(?:%|[x×])?", text))
    quantities.update(re.sub(r"\s+", "", value).casefold() for value in re.findall(
        r"[-+]?\d+(?:[.,]\d+)*\s*(?:ms|seconds?|secs?|minutes?|mins?|hours?|GB|MB|GiB|MiB|Hz|GHz|MHz|밀리초|초|분)(?![A-Za-z])", text
    ))
    identifiers = technical_identifiers(text)
    qualifications = set(re.findall(
        r"\b(?:not|never|only|except|without|higher|lower|increase\w*|decrease\w*|"
        r"improve\w*|worse|more|less|up to|at least|at most)\b", text.casefold()
    ))
    return quantities | identifiers | qualifications


def technical_identifiers(text):
    """Recognize acronyms, CamelCase names and capitalized model identifiers."""
    return set(re.findall(
        r"\b(?:[A-Z]{2,}[A-Za-z0-9_-]*|[A-Za-z]+[-_]\d+[A-Za-z0-9_-]*|"
        r"[A-Z][A-Za-z0-9]*[-_][A-Z0-9][A-Za-z0-9_-]*|"
        r"[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+)\b", text
    ))


def normalize_text(text):
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def text_shingles(text, size=3):
    tokens = normalize_text(text).split()
    if len(tokens) < size:
        return set(tokens)
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def jaccard_similarity(left, right):
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def evidence_object(item, rank):
    """Convert one scored chunk into the stable UI evidence contract."""
    chunk = item["chunk"]
    section = chunk.get("section")
    source_ranks = {
        name: int(value)
        for name, value in item.get("source_ranks", {}).items()
    }
    return {
        "evidence_id": f"ev-{rank:02d}-{chunk['chunk_id']}",
        "rank": rank,
        "paper_id": chunk["paper_id"],
        "page": int(chunk["page"]),
        "section": section,
        "chunk_id": chunk["chunk_id"],
        "source_page_id": chunk.get(
            "source_page_id", f"{chunk['paper_id']}-p{int(chunk['page']):03d}"
        ),
        "text": chunk["text"],
        "char_count": int(chunk.get("char_count", len(chunk["text"]))),
        "locator": {
            "page_label": f"p. {int(chunk['page'])}",
            "section": section,
            "chunk_id": chunk["chunk_id"],
        },
        "scores": {
            "reranker": float(item.get("reranker_score", item["score"])),
            "hybrid": float(item.get("retrieval_score", item["score"])),
            "pre_rerank_rank": int(item.get("pre_rerank_rank", item["rank"])),
            "source_ranks": source_ranks,
        },
        "source": {
            "section_index": int(chunk.get("section_index", 0)),
            "chunk_index": int(chunk.get("chunk_index", 0)),
        },
    }


def select_evidence(
    candidates,
    top_k=5,
    max_per_page=2,
    near_duplicate_threshold=0.85,
    question=None,
):
    """Select diverse evidence in reranker order using deterministic rules.

    Duplicate chunk IDs and normalized text are always removed. Highly similar
    chunks are removed using token 3-shingle Jaccard similarity only when their
    quantities and technical identifiers also match. The default page cap is
    retained for ordinary questions. Multi-part questions may use up to top_k
    chunks from a page, because one page can contain an entire requested list.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if max_per_page <= 0:
        raise ValueError("max_per_page must be greater than zero")
    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between zero and one")

    selected = []
    seen_chunk_ids = set()
    seen_texts = set()
    selected_shingles = []
    selected_facts = []
    page_counts = Counter()
    skipped = Counter()
    kinds = question_kinds(question)
    effective_page_cap = top_k if kinds else max_per_page

    # A fixed-width PDF chunk can end halfway through an answer-bearing causal
    # or defining statement. Retain its overlapping continuation when that
    # chunk is already in the reranked pool; never fetch unseen source text.
    ordered_candidates = list(candidates)
    continuation_ids = set()
    if kinds & {"causal", "definition"}:
        for anchor in candidates[:top_k]:
            chunk = anchor["chunk"]
            tail = re.split(r"(?<=[.!?])\s+", chunk["text"])[-1]
            if (re.search(r"[.!?]\s*$", tail)
                or not re.search(r"\b(?:because|due to|reason\w*|issue\w*|conjectur\w*|"
                                 r"caus\w*|therefore|means?|defined?|refers?)\b", tail, re.I)):
                continue
            section_index = chunk.get("section_index")
            chunk_index = chunk.get("chunk_index")
            if section_index is None or chunk_index is None:
                continue
            continuation = next((item for item in candidates if
                item["chunk"]["paper_id"] == chunk["paper_id"]
                and item["chunk"]["page"] == chunk["page"]
                and item["chunk"].get("section_index") == section_index
                and item["chunk"].get("chunk_index") == chunk_index + 1
                and len(chunk["text"].strip()) >= 40
                and chunk["text"].strip()[-40:] in item["chunk"]["text"][:300]
            ), None)
            if continuation is None:
                continue
            continuation_id = continuation["chunk"]["chunk_id"]
            ordered_candidates = [item for item in ordered_candidates
                                  if item["chunk"]["chunk_id"] != continuation_id]
            anchor_index = next(index for index, item in enumerate(ordered_candidates)
                                if item["chunk"]["chunk_id"] == chunk["chunk_id"])
            ordered_candidates.insert(anchor_index + 1, continuation)
            continuation_ids.add(continuation_id)
            if len(continuation_ids) == 2:
                break

    # Reserve already-reranked chunks for explicitly requested secondary
    # aspects (such as a flip condition). A repeated model name cannot replace
    # an absent requested aspect merely by scoring higher.
    facets = requested_source_facets(question)
    context_facets = [facet for facet in facets if facet.get("answer_context_only")]
    facets = [facet for facet in facets if not facet.get("answer_context_only")]
    coverage_candidates = ordered_candidates[:top_k]
    uncovered = [facet for facet in facets if not any(
        _facet_present(facet, item["chunk"]["text"])
        for item in coverage_candidates
    )]
    reserved_ids = []
    for facet in uncovered:
        supplemental = next((item for item in ordered_candidates if
                             _facet_present(facet, item["chunk"]["text"])), None)
        if supplemental is None:
            continue
        supplemental_id = supplemental["chunk"]["chunk_id"]
        if supplemental_id not in reserved_ids:
            reserved_ids.append(supplemental_id)
    for offset, supplemental_id in enumerate(reserved_ids[:max(0, top_k - 1)], start=1):
        supplemental = next(item for item in ordered_candidates if item["chunk"]["chunk_id"] == supplemental_id)
        ordered_candidates = [item for item in ordered_candidates if item["chunk"]["chunk_id"] != supplemental_id]
        ordered_candidates.insert(offset, supplemental)

    for item in ordered_candidates:
        chunk = item["chunk"]
        chunk_id = chunk["chunk_id"]
        normalized = normalize_text(chunk["text"])
        shingles = text_shingles(chunk["text"])
        facts = fact_tokens(chunk["text"])
        page_key = (chunk["paper_id"], int(chunk["page"]))

        if chunk_id in seen_chunk_ids:
            skipped["duplicate_chunk_id"] += 1
            continue
        seen_chunk_ids.add(chunk_id)
        text_key = (normalized, frozenset(facts))
        if text_key in seen_texts:
            skipped["duplicate_text"] += 1
            continue
        if any(
            jaccard_similarity(shingles, prior) >= near_duplicate_threshold
            and facts == prior_facts
            for prior, prior_facts in zip(selected_shingles, selected_facts)
        ):
            skipped["near_duplicate"] += 1
            continue
        if page_counts[page_key] >= effective_page_cap:
            skipped["page_cap"] += 1
            continue

        seen_texts.add(text_key)
        selected_shingles.append(shingles)
        selected_facts.append(facts)
        page_counts[page_key] += 1
        selected.append(evidence_object(item, len(selected) + 1))
        if len(selected) == top_k:
            break

    # Additional plain task definitions support answer construction without
    # replacing any ranked retrieval evidence. Every source is already in the
    # same reranked pool, and its original reranker/pre-rerank scores survive.
    context_supplements = []
    context_origins = []
    selected_chunk_ids = {item["chunk_id"] for item in selected}
    selected_paper_ids = {item["paper_id"] for item in selected}
    for facet in context_facets:
        if any(_facet_present(facet, item["text"]) for item in selected):
            continue
        supplement = next((item for item in candidates if
                           item["chunk"]["paper_id"] in selected_paper_ids
                           and item["chunk"]["chunk_id"] not in selected_chunk_ids
                           and _facet_present(facet, item["chunk"]["text"])), None)
        if supplement is not None:
            extra = evidence_object(supplement, len(selected) + len(context_supplements) + 1)
            context_supplements.append(extra)
            context_origins.append({
                "evidence_id": extra["evidence_id"], "chunk_id": extra["chunk_id"],
                "candidate_rank": int(supplement["rank"]), "facet": facet["name"],
                "pool": "existing_reranked_candidates", "source": "full_chunk",
                "source_text_sha256": hashlib.sha256(extra["text"].encode("utf-8")).hexdigest(),
            })
            selected_chunk_ids.add(supplement["chunk"]["chunk_id"])
        if len(context_supplements) == 1:
            break

    return selected, {
        "input_candidates": len(candidates),
        "selected": len(selected),
        "answer_context_supplements": context_supplements,
        "answer_context_supplement_origins": context_origins,
        "skipped": {
            "duplicate_chunk_id": skipped["duplicate_chunk_id"],
            "duplicate_text": skipped["duplicate_text"],
            "near_duplicate": skipped["near_duplicate"],
            "page_cap": skipped["page_cap"],
        },
        "policy": {
            "top_k": top_k,
            "max_per_page": max_per_page,
            "effective_max_per_page": effective_page_cap,
            "question_kinds": sorted(kinds),
            "overlapping_continuations": sorted(continuation_ids),
            "requested_aspect_supplements": reserved_ids,
            "near_duplicate_metric": "token_3_shingle_jaccard",
            "near_duplicate_threshold": near_duplicate_threshold,
        },
    }
