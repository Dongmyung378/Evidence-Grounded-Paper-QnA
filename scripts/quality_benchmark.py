"""Paper-disjoint, bilingual evaluation of complete final Q&A outputs.

Synthetic fixtures and disjoint real-paper extracted text are evaluated
separately. Automatic contract, refusal and marker scores are not semantic accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "quality_benchmark.json"
QUESTION_TYPES = {"fact", "numeric", "list", "comparison", "limitation"}
ANSWERABILITY = {"answerable", "unsupported", "false_premise"}
LANGUAGES = {"en", "ko"}
SPLITS = {"dev", "holdout"}
REFUSALS = {
    "en": "The answer cannot be verified from the provided paper evidence.",
    "ko": "제공된 논문 근거에서 답을 확인할 수 없습니다.",
}
HANGUL = re.compile(r"[가-힣]")
EVIDENCE_ID = re.compile(r"^ev-[0-9]{2}-.+$")
RUNTIME_CONFIGS = ["config/production_retrieval.json", "config/abstention.json", "config/answer_generation.json",
                   "config/answerability_policy.json", "config/translation.json", "config/translation_glossary.json",
                   "config/runtime_qna.json", "config/model_lock.json", "config/answer_output_schema.json",
                   "config/generation_runtime.json"]
IMPLEMENTATIONS = ["scripts/qna_pipeline.py", "scripts/production_retrieval.py", "scripts/candidate_evidence_pipeline.py",
                   "scripts/evidence_selector.py", "scripts/grounded_generation.py", "scripts/abstention_policy.py",
                   "scripts/local_translation.py", "scripts/model_lock.py", "scripts/dense_retrieval.py",
                   "scripts/reranker.py", "scripts/answerability_policy.py", "scripts/quality_benchmark.py",
                   "scripts/bm25_retrieval.py", "scripts/hybrid_retrieval.py", "scripts/retrieval_common.py",
                   "scripts/grounded_answer_contract.py", "scripts/answer_quality.py", "scripts/local_llm.py",
                   "app/question_engine.py", "scripts/ingest_papers.py"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def source_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Benchmark source must stay inside the project: {value}")
    return path


def marker_present(answer: str, alternative: str) -> bool:
    """Literal, normalized token coverage only, never an entailment judgment."""
    term = normalized(alternative)
    start = r"(?<![\w.])" if term[0].isalnum() else ""
    particles = r"(?:(?:이었습니다|입니다|이었다|였습니다|이다|이며|이고|인|일|하여|하는|하지|되어|되며|된|한|에서|에게|으로|부터|까지|처럼|보다|은|는|을|를|이|가|에|로|와|과|도|만|의|개|가지|배|초|명|회|층)(?:는|도|만)?)?"
    end = particles + (r"(?!\w|\.\d)" if term[-1].isdigit() else r"(?!\w)") if term[-1].isalnum() else ""
    if HANGUL.search(term):
        # Korean suffixes attach without whitespace. These are stem-coverage
        # signals; accepting a stem deliberately does not prove its polarity.
        end = r"(?![a-z0-9])"
    pattern = r"[\s-]*".join(re.escape(part) for part in re.split(r"[\s-]+", term))
    return re.search(start + pattern + end, normalized(answer)) is not None


def runtime_hashes() -> tuple[dict, dict]:
    # Optional new runtime assets remain backwards compatible until installed.
    configs = {path: digest(ROOT / path) for path in RUNTIME_CONFIGS if (ROOT / path).is_file()}
    implementations = {path: digest(ROOT / path) for path in IMPLEMENTATIONS}
    return configs, implementations


def verify_source_pdfs(manifest: dict, folder: Path, split: str) -> dict:
    """Match locally obtained publisher PDFs to the frozen original byte hashes."""
    checked = []
    for paper in manifest["papers"]:
        if split != "all" and paper["split"] != split:
            continue
        source = read_json(source_path(paper["source_path"]))
        if source.get("source_kind") != "publisher_pdf_extracted_text":
            raise ValueError("verify-pdfs requires the real-paper benchmark config")
        filename = source["attribution"]["pdf_url"].rsplit("/", 1)[-1]
        path = folder / filename
        if not path.is_file() or digest(path) != source["pdf_sha256"]:
            raise ValueError(f"Original publisher PDF missing or byte hash mismatch: {paper['paper_id']} ({filename})")
        checked.append({"paper_id": paper["paper_id"], "filename": filename,
                        "pdf_sha256": source["pdf_sha256"], "verified": True})
    return {"benchmark_id": manifest["benchmark_id"], "scope": "original_pdf_byte_identity_only",
            "split": split, "checked_pdfs": checked, "independent_human_review": False}


def runtime_freeze(config: dict, manifest: dict) -> dict:
    configs, implementations = runtime_hashes()
    return {"schema_version": 1, "benchmark_id": config["benchmark_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runtime_config_sha256": configs, "implementation_sha256": implementations,
            "harness_sha256": digest(Path(__file__)),
            "frozen_inputs": read_json(source_path(config["freeze"]))["sha256"],
            "paper_sources_sha256": {p["paper_id"]: p["source_sha256"] for p in manifest["papers"]},
            "claim_boundary": "Settings receipt only. This does not attest independent review or model-training disjointness."}


def preserve_runtime_sources(receipt: dict, output: Path) -> dict:
    """Keep exact measured code and settings beside the receipt, without replacement."""
    expected = {**receipt["runtime_config_sha256"], **receipt["implementation_sha256"]}
    folder = output.resolve().parent / (output.stem + "-sources")
    assert_writable_output(folder)
    payloads = {}
    for relative, wanted in expected.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or relative_path.drive or ".." in relative_path.parts:
            raise ValueError("Runtime snapshot keys must be relative project paths")
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT.resolve()):
            raise ValueError("Runtime source must stay inside the project")
        contents = source.read_bytes()
        if hashlib.sha256(contents).hexdigest() != wanted:
            raise ValueError("Runtime source changed before snapshot: " + relative)
        payloads[relative] = contents
    if folder.exists():
        existing = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
        if existing != set(expected) or any(digest(folder / relative) != wanted for relative, wanted in expected.items()):
            raise ValueError("Existing runtime snapshot differs; use a new receipt path")
    else:
        folder.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="runtime-snapshot-", dir=folder.parent) as temporary:
            staging = Path(temporary) / "sources"
            for relative, contents in payloads.items():
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents)
            if any(digest(ROOT / relative) != wanted for relative, wanted in expected.items()):
                raise ValueError("Runtime source changed during snapshot")
            if (folder.resolve().parent != output.resolve().parent
                    or not staging.resolve().is_relative_to(output.resolve().parent)):
                raise ValueError("Runtime snapshot targets must stay beside their receipt")
            staging.rename(folder)
    return {"directory": folder.name, "sha256": expected, "byte_exact": True}


def validate_runtime_freeze(config: dict, manifest: dict, path: Path | None) -> dict:
    if path is None:
        raise ValueError("New real-paper holdout requires freeze-run followed by --runtime-freeze; fix settings using DEV only")
    receipt = read_json(path)
    expected = runtime_freeze(config, manifest)
    for field in ("schema_version", "benchmark_id", "runtime_config_sha256", "implementation_sha256",
                  "harness_sha256", "frozen_inputs", "paper_sources_sha256"):
        if receipt.get(field) != expected[field]:
            raise ValueError(f"Runtime freeze differs from current {field}; do not tune using holdout outputs")
    snapshot = receipt.get("source_snapshot")
    if snapshot is not None:
        directory = (path.resolve().parent / snapshot.get("directory", "")).resolve()
        if not directory.is_relative_to(path.resolve().parent) or directory == path.resolve().parent:
            raise ValueError("Runtime snapshot path must stay beside its receipt")
        wanted = {**expected["runtime_config_sha256"], **expected["implementation_sha256"]}
        if snapshot.get("sha256") != wanted or snapshot.get("byte_exact") is not True:
            raise ValueError("Runtime snapshot provenance differs from the receipt")
        for relative, value in wanted.items():
            target = directory / relative
            if not target.is_file() or digest(target) != value:
                raise ValueError("Runtime snapshot missing or modified: " + relative)
    return {"path": str(path), "sha256": digest(path), "created_at": receipt.get("created_at")}


def validate_dataset(config: dict, manifest: dict, questions: list[dict], rubric: dict) -> dict:
    """Reject paper, translation-group and exact-question leakage before scoring."""
    if config.get("schema_version") != 1 or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported quality benchmark schema")
    if config.get("benchmark_id") != manifest.get("benchmark_id"):
        raise ValueError("Benchmark ID mismatch")
    if rubric.get("schema_version") != 1 or rubric.get("frozen") is not True:
        raise ValueError("The scoring rubric must be versioned and frozen")
    if not questions:
        raise ValueError("The benchmark requires questions")
    papers = manifest.get("papers", [])
    paper_ids = [paper.get("paper_id") for paper in papers]
    if any(not isinstance(item, str) or not item for item in paper_ids) or len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Paper IDs must be unique non-empty strings")
    paper_map = {paper["paper_id"]: paper for paper in papers}
    source_hashes = defaultdict(set)
    article_ids = set()
    for paper in papers:
        if paper.get("split") not in SPLITS:
            raise ValueError("Each paper must belong to dev or holdout")
        path = source_path(paper["source_path"])
        if digest(path) != paper.get("source_sha256"):
            raise ValueError(f"Paper source hash mismatch: {paper['paper_id']}")
        source_hashes[paper["source_sha256"]].add(paper["split"])
        pages = paper.get("pages")
        if not isinstance(pages, list) or not pages:
            raise ValueError("Each paper must declare its available page numbers")
        if len(pages) != len(set(pages)) or any(type(page) is not int or page < 1 for page in pages):
            raise ValueError("Paper page numbers must be unique positive integers")
        if manifest.get("purpose") == "real_paper_text_evaluation":
            article = paper.get("arxiv_id")
            if not isinstance(article, str) or not re.fullmatch(r"\d{4}\.\d{4,5}", article) or article in article_ids:
                raise ValueError("Real papers require unique version-independent arXiv identities")
            article_ids.add(article)
            source = read_json(path)
            if source.get("source_kind") != "publisher_pdf_extracted_text" or source.get("synthetic") is not False:
                raise ValueError("Real evaluation sources must be identifiable publisher PDF text")
            if [page["page"] for page in source.get("pages", [])] != pages:
                raise ValueError("Extracted source pages differ from the manifest")
            attribution = source.get("attribution", {})
            for field in ("title", "authors", "publisher_url", "pdf_url", "arxiv_url", "doi", "license_basis_url"):
                if not attribution.get(field):
                    raise ValueError(f"Real-paper source attribution is missing {field}")
            if attribution.get("license_id") != "CC-BY-4.0" or not re.fullmatch(r"[0-9a-f]{64}", source.get("pdf_sha256", "")):
                raise ValueError("Real-paper PDF hash and checked redistribution basis are required")
    if manifest.get("purpose") == "real_paper_text_evaluation":
        exclusion_path = source_path(config["corpus_exclusion_manifest"])
        with exclusion_path.open(encoding="utf-8-sig", newline="") as handle:
            existing = {re.search(r"/(\d{4}\.\d{4,5})", row.get("source_url", "")).group(1)
                        for row in csv.DictReader(handle) if re.search(r"/(\d{4}\.\d{4,5})", row.get("source_url", ""))}
        if article_ids & existing:
            raise ValueError("Real-paper DEV/HOLDOUT overlaps the existing project corpus")
    if any(len(splits) > 1 for splits in source_hashes.values()):
        raise ValueError("Identical paper sources cannot cross dev and holdout")
    if len(source_hashes) != len(papers):
        raise ValueError("Duplicate paper source bytes cannot be counted as distinct papers")

    case_ids, question_texts, groups = set(), set(), defaultdict(list)
    for row in questions:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("Case IDs must be unique non-empty strings")
        case_ids.add(case_id)
        paper = paper_map.get(row.get("paper_id"))
        if paper is None or row.get("split") != paper["split"]:
            raise ValueError(f"Paper split leakage or unknown paper: {case_id}")
        if row.get("language") not in LANGUAGES or row.get("question_type") not in QUESTION_TYPES:
            raise ValueError(f"Unknown question language or category: {case_id}")
        if row.get("answerability") not in ANSWERABILITY:
            raise ValueError(f"Unknown answerability: {case_id}")
        text = row.get("question")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Question text is required: {case_id}")
        key = normalized(text)
        if key in question_texts:
            raise ValueError(f"Duplicate normalized question text: {case_id}")
        question_texts.add(key)
        detected_language = "ko" if HANGUL.search(text) else "en"
        if detected_language != row["language"]:
            raise ValueError(f"Question language mismatch: {case_id}")
        meaning_id = row.get("meaning_id")
        if not isinstance(meaning_id, str) or not meaning_id:
            raise ValueError(f"Meaning group ID is required: {case_id}")
        gold_pages = row.get("gold_pages")
        if not isinstance(gold_pages, list) or len(gold_pages) != len(set(gold_pages)):
            raise ValueError(f"Gold pages must be a unique list: {case_id}")
        if any(type(page) is not int or page not in paper["pages"] for page in gold_pages):
            raise ValueError(f"Gold page is not present in the paper: {case_id}")
        if row["answerability"] == "answerable" and not gold_pages:
            raise ValueError(f"Answerable cases require supporting pages: {case_id}")
        if row["answerability"] != "answerable" and gold_pages:
            raise ValueError(f"Refusal cases must not have supporting gold pages: {case_id}")
        markers = row.get("required_markers", [])
        if manifest.get("purpose") == "real_paper_text_evaluation" and row["answerability"] == "answerable" and not markers:
            raise ValueError(f"Real-paper answerable cases require explicit coverage markers: {case_id}")
        seen_markers = set()
        for marker in markers:
            if not isinstance(marker, dict) or not isinstance(marker.get("id"), str) or marker["id"] in seen_markers:
                raise ValueError(f"Coverage markers need unique IDs: {case_id}")
            seen_markers.add(marker["id"])
            alternatives = marker.get("any_of")
            if not isinstance(alternatives, list) or not alternatives or any(not isinstance(term, str) or not term.strip() for term in alternatives):
                raise ValueError(f"Coverage markers need non-empty literal alternatives: {case_id}")
            if not any(marker_present(row.get("reference_answer", ""), term) for term in alternatives):
                raise ValueError(f"Reference answer omits its own required marker: {case_id} ({marker['id']})")
        if row["answerability"] != "answerable" and markers:
            raise ValueError(f"Refusal cases cannot declare answer coverage markers: {case_id}")
        groups[meaning_id].append(row)
    for meaning_id, pair in groups.items():
        if len(pair) != 2 or {row["language"] for row in pair} != LANGUAGES:
            raise ValueError(f"Each meaning group requires exactly one EN and one KO case: {meaning_id}")
        for field in ("paper_id", "split", "question_type", "answerability", "gold_pages"):
            if pair[0][field] != pair[1][field]:
                raise ValueError(f"Translation group crosses papers or labels: {meaning_id} ({field})")
        marker_ids = [{marker["id"] for marker in row.get("required_markers", [])} for row in pair]
        if marker_ids[0] != marker_ids[1]:
            raise ValueError(f"Translation group coverage markers differ: {meaning_id}")
    for split in SPLITS:
        cases = [row for row in questions if row["split"] == split]
        if {row["question_type"] for row in cases} != QUESTION_TYPES:
            raise ValueError(f"Every split must cover all five question types: {split}")
        if {row["answerability"] for row in cases} != ANSWERABILITY:
            raise ValueError(f"Every split must cover all three answerability classes: {split}")
    if set(paper_ids) != {row["paper_id"] for row in questions}:
        raise ValueError("Every declared paper must have questions")
    return {
        "papers": len(papers), "outputs_required": len(questions),
        "meaning_groups": len(groups),
        "groups_by_split": dict(Counter(pair[0]["split"] for pair in groups.values())),
        "paper_disjoint": True, "translation_groups_disjoint": True,
        "purpose": manifest["purpose"],
        "independent_human_review": False,
        "statistical_independence_assumed": False,
    }


def load_dataset(config_path: Path = DEFAULT_CONFIG) -> tuple[dict, dict, list[dict], dict]:
    config = read_json(config_path)
    manifest = read_json(source_path(config["manifest"]))
    questions = read_jsonl(source_path(config["questions"]))
    rubric = read_json(source_path(config["rubric"]))
    freeze = read_json(source_path(config["freeze"]))
    expected = {
        config["manifest"]: digest(source_path(config["manifest"])),
        config["questions"]: digest(source_path(config["questions"])),
        config["rubric"]: digest(source_path(config["rubric"])),
    }
    expected.update({value: digest(source_path(value)) for value in config.get("extra_frozen_inputs", [])})
    if freeze.get("sha256") != expected or freeze.get("benchmark_id") != config["benchmark_id"]:
        raise ValueError("Frozen benchmark inputs changed; review and version the benchmark before running")
    validate_dataset(config, manifest, questions, rubric)
    return config, manifest, questions, rubric


def final_output(record: dict) -> tuple[dict, list[dict], dict]:
    """Read an internal pipeline result or a public /question response, never a retrieval decision."""
    output = record.get("output")
    if not isinstance(output, dict):
        raise ValueError("A complete final output object is required")
    if "response" in output:
        payload = output["response"]
        evidence = output.get("cited_evidence", [])
        runtime = output.get("pipeline", {})
    else:
        payload = {key: output.get(key) for key in ("answer", "sufficiency", "abstention_reason")}
        evidence = output.get("evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(item, dict) for item in evidence):
            raise ValueError("Final evidence must be a list of objects")
        payload["evidence_ids"] = [item.get("evidence_id") for item in evidence]
        runtime = output.get("runtime", {})
    if not isinstance(payload, dict) or not isinstance(evidence, list) or not isinstance(runtime, dict):
        raise ValueError("Malformed final output")
    return payload, evidence, runtime


def inspect_final_output(case: dict, record: dict) -> dict:
    """Score observable final behavior; semantic correctness needs a separate review."""
    result = {
        "case_id": case["case_id"], "meaning_id": case["meaning_id"],
        "split": case["split"], "language": case["language"],
        "question_type": case["question_type"], "answerability": case["answerability"],
        "valid_contract": False, "final_refusal": False, "final_answer": False,
        "all_required_pages_cited": False, "citation_locator_recall": None,
        "fallback_used": False, "abstention_source": None, "errors": [],
        "required_marker_count": len(case.get("required_markers", [])),
        "required_markers_present": [], "all_required_markers_present": None,
    }
    try:
        payload, evidence, runtime = final_output(record)
        output = record["output"]
        expected_paper = record.get("runtime_paper_id") or case["paper_id"]
        if output.get("paper_id") != expected_paper:
            raise ValueError("Final output belongs to another paper or lacks its paper identity")
        echoed_question = output.get("query") if "response" in output else output.get("question")
        if echoed_question != case["question"] or output.get("question_language") != case["language"]:
            raise ValueError("Final output question or language identity does not match the benchmark case")
        result["fallback_used"] = runtime.get("fallback_used") is True
        result["abstention_source"] = runtime.get("abstention_source")
        if set(payload) != {"answer", "evidence_ids", "sufficiency", "abstention_reason"}:
            raise ValueError("Final payload fields do not match the answer contract")
        answer, ids = payload["answer"], payload["evidence_ids"]
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 4000:
            raise ValueError("Final answer must contain text and stay within the contract limit")
        if ("ko" if HANGUL.search(answer) else "en") != case["language"]:
            raise ValueError("Final answer language differs from the question")
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ValueError("Final evidence IDs must be a list of strings")
        if len(ids) > 5 or len(set(ids)) != len(ids) or any(not EVIDENCE_ID.fullmatch(item) for item in ids):
            raise ValueError("Final evidence IDs violate the citation contract")
        if any(not isinstance(item, dict) for item in evidence):
            raise ValueError("Cited evidence must contain objects")
        evidence_ids = [item.get("evidence_id") for item in evidence]
        if len(set(evidence_ids)) != len(evidence_ids) or set(evidence_ids) != set(ids):
            raise ValueError("Final citations must resolve exactly to the returned evidence")
        for item in evidence:
            if type(item.get("page")) is not int or item["page"] < 1:
                raise ValueError("Cited evidence requires a positive page number")
            if not isinstance(item.get("text"), str) or not item["text"].strip():
                raise ValueError("Cited evidence requires source text")
            allowed_papers = {case["paper_id"], record.get("runtime_paper_id")}
            if item.get("paper_id") is not None and item["paper_id"] not in allowed_papers:
                raise ValueError("Cited evidence belongs to another paper")
            chunk_id = item["evidence_id"].split("-", 2)[2]
            if not chunk_id.startswith(expected_paper + "-"):
                raise ValueError("Cited evidence ID is scoped to another paper")
            locator = item.get("locator")
            if not isinstance(locator, dict) or locator.get("chunk_id") != chunk_id:
                raise ValueError("Citation locator must resolve to its exact scoped chunk ID")
            if item.get("chunk_id", chunk_id) != chunk_id:
                raise ValueError("Cited chunk ID differs from the evidence ID")
        if payload["sufficiency"] == "sufficient":
            if not ids or payload["abstention_reason"] is not None:
                raise ValueError("A sufficient final answer requires citations and no refusal reason")
            if normalized(answer) in {normalized(value) for value in REFUSALS.values()}:
                raise ValueError("A sufficient answer cannot contain the canonical refusal")
            if normalized(answer) == normalized(case["question"]) or normalized(answer) == "string":
                raise ValueError("A sufficient output cannot repeat the question or a placeholder")
            result["final_answer"] = True
        elif payload["sufficiency"] == "insufficient":
            reason = payload["abstention_reason"]
            if ids or not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
                raise ValueError("A refused final answer requires no citations and a non-empty reason")
            if answer.strip() != REFUSALS[case["language"]]:
                raise ValueError("The final text must be the canonical refusal, not an unsupported answer")
            result["final_refusal"] = True
        else:
            raise ValueError("Final sufficiency must be sufficient or insufficient")
        result["valid_contract"] = True
        markers = case.get("required_markers", [])
        if markers:
            result["required_markers_present"] = [marker["id"] for marker in markers
                                                    if any(marker_present(answer, term) for term in marker["any_of"])]
            result["all_required_markers_present"] = (result["final_answer"] and
                                                       len(result["required_markers_present"]) == len(markers))
        if case["gold_pages"]:
            cited_pages = {item["page"] for item in evidence}
            result["citation_locator_recall"] = len(cited_pages & set(case["gold_pages"])) / len(case["gold_pages"])
            result["all_required_pages_cited"] = set(case["gold_pages"]).issubset(cited_pages)
    except (ValueError, TypeError, KeyError) as error:
        result["errors"].append(str(error))
        result["final_answer"] = result["final_refusal"] = False
    return result


def ratio(passed: int, total: int) -> float | None:
    return round(passed / total, 4) if total else None


def summarize(records: list[dict], *, grouped: bool) -> dict:
    buckets = defaultdict(list)
    for row in records:
        buckets[row["meaning_id"] if grouped else row["case_id"]].append(row)
    units = list(buckets.values())
    answerable = [unit for unit in units if unit[0]["answerability"] == "answerable"]
    unsupported = [unit for unit in units if unit[0]["answerability"] == "unsupported"]
    false_premise = [unit for unit in units if unit[0]["answerability"] == "false_premise"]
    marked = [unit for unit in answerable if any(row["required_marker_count"] for row in unit)]
    refused = lambda unit: all(row["final_refusal"] for row in unit)
    answered = lambda unit: all(row["final_answer"] for row in unit)
    return {
        "units": len(units), "unit": "meaning_group" if grouped else "language_output",
        "valid_final_contract_rate": ratio(sum(all(row["valid_contract"] for row in unit) for unit in units), len(units)),
        "answerable_groups_or_outputs": len(answerable),
        "answerable_retention": ratio(sum(answered(unit) for unit in answerable), len(answerable)),
        "false_refusal_rate": ratio(sum(any(row["final_refusal"] for row in unit) for unit in answerable), len(answerable)),
        "unsupported_groups_or_outputs": len(unsupported),
        "unsupported_final_refusal_rate": ratio(sum(refused(unit) for unit in unsupported), len(unsupported)),
        "false_premise_groups_or_outputs": len(false_premise),
        "false_premise_final_refusal_rate": ratio(sum(refused(unit) for unit in false_premise), len(false_premise)),
        "unsupported_answer_rate": ratio(sum(any(row["final_answer"] for row in unit) for unit in unsupported + false_premise), len(unsupported + false_premise)),
        "all_required_pages_cited_rate": ratio(sum(all(row["all_required_pages_cited"] for row in unit) for unit in answerable), len(answerable)),
        "invalid_outputs": sum(not row["valid_contract"] for unit in units for row in unit),
        "fallback_outputs": sum(row["fallback_used"] for unit in units for row in unit),
        "marker_checked_groups_or_outputs": len(marked),
        "all_required_markers_present_rate": ratio(sum(all(row["all_required_markers_present"] is True for row in unit) for unit in marked), len(marked)),
    }


def evaluate(config: dict, manifest: dict, questions: list[dict], predictions: dict) -> dict:
    if predictions.get("schema_version") != 1 or predictions.get("benchmark_id") != config["benchmark_id"]:
        raise ValueError("Predictions must identify this benchmark and schema")
    execution = predictions.get("execution")
    if not isinstance(execution, dict) or execution.get("kind") not in {"stub_fixture", "saved_pipeline_outputs", "api_pipeline", "local_pipeline"}:
        raise ValueError("Predictions must declare a supported execution kind")
    selected_split = execution.get("split", "all")
    if selected_split not in SPLITS | {"all"}:
        raise ValueError("Execution split must be dev, holdout, or all")
    if selected_split != "all":
        questions = [row for row in questions if row["split"] == selected_split]
    inputs = predictions.get("records")
    if not isinstance(inputs, list) or any(not isinstance(row, dict) for row in inputs):
        raise ValueError("Prediction records must be a list of objects")
    identifiers = [row.get("case_id") for row in inputs]
    if any(not isinstance(item, str) for item in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Prediction case IDs must be unique strings")
    expected = {row["case_id"] for row in questions}
    if set(identifiers) != expected:
        raise ValueError(f"Predictions must cover every case exactly; missing={sorted(expected - set(identifiers))}, extra={sorted(set(identifiers) - expected)}")
    prediction_map = {row["case_id"]: row for row in inputs}
    records = [inspect_final_output(case, prediction_map[case["case_id"]]) for case in questions]
    return {
        "schema_version": 2, "benchmark_id": config["benchmark_id"],
        "purpose": manifest["purpose"], "execution": execution,
        "claim_boundary": {
            "synthetic_fixture": manifest["purpose"] == "synthetic_integration_only",
            "real_paper_holdout_accuracy_claim": False,
            "automatic_semantic_accuracy": None,
            "independent_human_review": False,
            "third_party_validation": "not_attested",
            "citation_locator_metrics_are_not_semantic_support": True,
            "statistical_independence_assumed": False,
            "model_training_disjointness_verified": False,
            "lexical_marker_coverage_is_not_semantic_accuracy": True,
        },
        "counts": {"papers": len({row["paper_id"] for row in questions}), "language_outputs": len(records),
                   "meaning_groups": len({row["meaning_id"] for row in records})},
        "provenance": {"evaluator_sha256": digest(Path(__file__)),
                       "frozen_inputs": read_json(source_path(config["freeze"]))["sha256"],
                       "paper_sources_sha256": {paper["paper_id"]: paper["source_sha256"] for paper in manifest["papers"]},
                       "rubric_id": read_json(source_path(config["rubric"]))["rubric_id"],
                       "scoring_implementation_sha256": digest(Path(__file__))},
        "metrics": summarize(records, grouped=True),
        "by_split": {split: summarize([row for row in records if row["split"] == split], grouped=True) for split in sorted(SPLITS)},
        "by_language": {language: summarize([row for row in records if row["language"] == language], grouped=False) for language in sorted(LANGUAGES)},
        "by_question_type": {kind: summarize([row for row in records if row["question_type"] == kind], grouped=True) for kind in sorted(QUESTION_TYPES)},
        "records": records,
    }


def prediction_template(config: dict, questions: list[dict]) -> dict:
    return {
        "schema_version": 1, "benchmark_id": config["benchmark_id"],
        "execution": {"kind": "saved_pipeline_outputs", "status": "not_run",
                      "pipeline_revision": "REPLACE_WITH_EXACT_CODE_COMMIT_AND_MODEL_CONFIG_FINGERPRINT",
                      "operator": "not_attested"},
        "records": [{"case_id": row["case_id"], "output": None} for row in questions],
    }


def smoke_predictions(config: dict, manifest: dict, questions: list[dict]) -> dict:
    """Gold-driven stubs test scoring only and are explicitly labelled as such."""
    papers = {paper["paper_id"]: read_json(source_path(paper["source_path"])) for paper in manifest["papers"]}
    rows = []
    for case in questions:
        evidence = []
        if case["answerability"] == "answerable":
            for index, page_number in enumerate(case["gold_pages"], 1):
                page = next(page for page in papers[case["paper_id"]]["pages"] if page["page"] == page_number)
                chunk_id = f"{case['paper_id']}-p{page_number:03d}-c001"
                evidence.append({"evidence_id": f"ev-{index:02d}-{chunk_id}",
                                 "paper_id": case["paper_id"], "chunk_id": chunk_id,
                                 "locator": {"chunk_id": chunk_id, "page_label": f"p. {page_number}", "section": page["section"]}, **page})
            payload = {"answer": case["reference_answer"], "evidence_ids": [item["evidence_id"] for item in evidence],
                       "sufficiency": "sufficient", "abstention_reason": None}
        else:
            payload = {"answer": REFUSALS[case["language"]], "evidence_ids": [], "sufficiency": "insufficient",
                       "abstention_reason": "Authored fixture has no support for the requested claim."}
        rows.append({"case_id": case["case_id"], "output": {"paper_id": case["paper_id"], "query": case["question"],
                     "question_language": case["language"], "response": payload, "cited_evidence": evidence,
                     "pipeline": {"fallback_used": False, "abstention_source": "fixture" if not evidence else None}}})
    return {"schema_version": 1, "benchmark_id": config["benchmark_id"],
            "execution": {"kind": "stub_fixture", "gold_used_to_construct_outputs": True,
                          "pipeline_revision": "none: deterministic harness fixtures", "operator": "not_attested"}, "records": rows}


def run_api(config: dict, questions: list[dict], base_url: str, paper_map: dict, revision: str, timeout: float) -> dict:
    if set(paper_map) != {row["paper_id"] for row in questions}:
        raise ValueError("The runtime paper map must include exactly the benchmark papers")
    if any(not isinstance(value, str) or not re.fullmatch(r"paper-[0-9a-f]{32}", value) for value in paper_map.values()):
        raise ValueError("Runtime paper IDs must be IDs returned by the upload API")
    if len(set(paper_map.values())) != len(paper_map):
        raise ValueError("Distinct benchmark papers cannot map to one runtime paper")
    rows = []
    for case in questions:
        runtime_id = paper_map[case["paper_id"]]
        body = json.dumps({"paper_id": runtime_id, "question": case["question"]}, ensure_ascii=False).encode("utf-8")
        request = Request(base_url.rstrip("/") + "/question", data=body, headers={"Content-Type": "application/json"}, method="POST")
        record = {"case_id": case["case_id"], "runtime_paper_id": runtime_id, "output": None}
        try:
            with urlopen(request, timeout=timeout) as response:
                record["output"] = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            # Failed calls remain in the denominator and are never treated as refusals.
            record["execution_error"] = f"{type(error).__name__}: {error}"
        rows.append(record)
    return {"schema_version": 1, "benchmark_id": config["benchmark_id"],
            "execution": {"kind": "api_pipeline", "pipeline_revision": revision,
                          "executed_at": datetime.now(timezone.utc).isoformat(),
                          "gold_exposed_to_pipeline": False, "operator": "not_attested",
                          "python": platform.python_version(), "platform": platform.platform()}, "records": rows}


def fixture_chunks(manifest: dict, split: str) -> list[dict]:
    """Convert source pages into runtime chunks without using questions or gold."""
    from ingest_papers import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_text
    chunks = []
    for paper in manifest["papers"]:
        if split != "all" and paper["split"] != split:
            continue
        source = read_json(source_path(paper["source_path"]))
        for page in source["pages"]:
            number = page["page"]
            texts = [page["text"]]
            if source.get("source_kind") == "publisher_pdf_extracted_text":
                # Share production text boundaries and overlap. No gold pages,
                # questions or labels inform source chunk construction.
                texts = split_text(page["text"], DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            for index, text in enumerate(texts, 1):
                chunks.append({"paper_id": paper["paper_id"], "page": number,
                           "chunk_id": f"{paper['paper_id']}-p{number:03d}-c{index:03d}",
                           "source_page_id": f"{paper['paper_id']}-p{number:03d}",
                           "text": text, "section": page["section"],
                           "char_count": len(text), "section_index": number - 1,
                           "chunk_index": index - 1})
    return chunks


def run_pipeline(config: dict, manifest: dict, questions: list[dict], *, split: str,
                 offline: bool, revision: str, cache_path: Path, device: str | None,
                 freeze_path: Path | None = None) -> dict:
    """Use the production retrieval, refusal, generation and translation models.

    Imports and downloads happen only in this explicit mode. The selected source
    text can be synthetic or extracted publisher-paper text. Both bypass PDF
    parsing: this is final-answer text evaluation, not an ingestion or visual test.
    """
    freeze_record = None
    if manifest.get("purpose") == "real_paper_text_evaluation" and split in {"holdout", "all"}:
        freeze_record = validate_runtime_freeze(config, manifest, freeze_path)
    config_hashes, implementation_hashes = runtime_hashes()
    from production_retrieval import ProductionRetrieval
    from qna_pipeline import GroundedQAPipeline
    from ingest_papers import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

    retrieval = ProductionRetrieval(chunks=fixture_chunks(manifest, split),
                                    embedding_cache_path=cache_path,
                                    local_files_only=offline,
                                    embedding_device=device, reranker_device=device)
    pipeline = GroundedQAPipeline(retrieval=retrieval, local_files_only=offline)
    rows = []
    for case in questions:
        if split != "all" and case["split"] != split:
            continue
        record = {"case_id": case["case_id"], "output": None}
        try:
            # Only the question and paper ID enter the pipeline. Gold stays in scoring.
            record["output"] = pipeline.ask(case["question"], case["paper_id"],
                                             question_language=case["language"], include_debug=True)
        except (ValueError, RuntimeError, OSError) as error:
            record["execution_error"] = f"{type(error).__name__}: {error}"
        rows.append(record)
        print(f"[{len(rows):02d}] {case['case_id']}: {'complete' if record['output'] is not None else 'failed'}", flush=True)
    current_configs, current_implementations = runtime_hashes()
    initial_hashes = {**config_hashes, **implementation_hashes}
    current_hashes = {**current_configs, **current_implementations}
    changed_sources = [path for path in sorted(initial_hashes.keys() | current_hashes.keys())
                       if initial_hashes.get(path) != current_hashes.get(path)]
    if changed_sources:
        raise ValueError("Runtime sources changed during evaluation; no report may be saved: " + ", ".join(changed_sources))
    package_versions = {}
    for package in ("torch", "transformers", "sentence-transformers", "sentencepiece", "numpy", "rank-bm25"):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = "not_installed"
    return {"schema_version": 1, "benchmark_id": config["benchmark_id"],
            "execution": {"kind": "local_pipeline", "split": split, "pipeline_revision": revision,
                          "executed_at": datetime.now(timezone.utc).isoformat(), "offline": offline,
                          "pdf_ingestion_exercised": False, "gold_exposed_to_pipeline": False,
                          "source_text_chunking": {"implementation": "scripts/ingest_papers.py:split_text",
                                                   "real_paper_chunk_size": DEFAULT_CHUNK_SIZE,
                                                   "real_paper_overlap": DEFAULT_CHUNK_OVERLAP,
                                                   "synthetic_sources": "one_chunk_per_authored_page"},
                          "operator": "not_attested", "python": platform.python_version(),
                          "platform": platform.platform(), "package_versions": package_versions,
                          "runtime_config_sha256": config_hashes,
                          "implementation_sha256": implementation_hashes,
                          "runtime_freeze": freeze_record,
                          "source_kind": manifest.get("provenance", {}).get("source_kind"),
                          "model_training_disjointness_verified": False},
            "records": rows}


def assert_writable_output(path: Path, protected_paths: set[Path] | None = None) -> None:
    """Refuse destructive output targets before executing a potentially long run."""
    destination = path.resolve()
    protected_directories = [ROOT / name for name in
                             ("config", "scripts", "app", "tests", "ui", "docs", "data/evaluation", "data/baselines",
                              "data/quality_benchmark/papers")]
    protected_directories.append(ROOT / "data/quality_benchmark/real_paper_sources")
    if any(destination == directory.resolve() or destination.is_relative_to(directory.resolve())
           for directory in protected_directories):
        raise ValueError("Output path would overwrite protected sources or frozen evaluation data")
    protected_files = {ROOT / "data" / "quality_benchmark" / name for name in
                       ("manifest.json", "questions.jsonl", "rubric.json", "freeze.json", "real_holdout_template.json")}
    protected_files.update(ROOT / "data" / "processed" / name for name in
                           ("dense_embeddings.npz", "frozen_dense_embeddings.npz"))
    protected_files.update((ROOT / "data/quality_benchmark").glob("real_paper_*.*"))
    protected_files.add(ROOT / "data/metadata/paper_manifest.csv")
    protected_files.update(protected_paths or set())
    if destination in {item.resolve() for item in protected_files}:
        raise ValueError("Output path would overwrite a frozen benchmark input or the prediction input")


def save(path: Path, payload: dict, protected_paths: set[Path] | None = None) -> None:
    assert_writable_output(path, protected_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    pdf_command = commands.add_parser("verify-pdfs")
    pdf_command.add_argument("--pdf-dir", type=Path, required=True)
    pdf_command.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    freeze_command = commands.add_parser("freeze-run")
    freeze_command.add_argument("--output", type=Path, required=True)
    template = commands.add_parser("template")
    template.add_argument("--output", type=Path, required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("evaluate")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    api = commands.add_parser("run-api")
    api.add_argument("--base-url", required=True)
    api.add_argument("--paper-map", type=Path, required=True)
    api.add_argument("--pipeline-revision", required=True)
    api.add_argument("--timeout", type=float, default=180)
    api.add_argument("--output", type=Path, required=True)
    api.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    api.add_argument("--runtime-freeze", type=Path)
    local = commands.add_parser("run")
    local.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    local.add_argument("--offline", action="store_true")
    local.add_argument("--device", choices=["cpu", "cuda"])
    local.add_argument("--pipeline-revision", default="working-tree: inspect runtime config hashes and git diff")
    local.add_argument("--embedding-cache", type=Path, default=ROOT / "data" / "processed" / "quality_benchmark_embeddings.npz")
    local.add_argument("--output", type=Path, required=True)
    local.add_argument("--runtime-freeze", type=Path)
    arguments = parser.parse_args()
    config, manifest, questions, rubric = load_dataset(arguments.config)
    protected_paths = {arguments.config, source_path(config["freeze"]), source_path(config["manifest"]),
                       source_path(config["questions"]), source_path(config["rubric"])}
    protected_paths.update(source_path(paper["source_path"]) for paper in manifest["papers"])
    protected_paths.update(source_path(value) for value in config.get("extra_frozen_inputs", []))
    if getattr(arguments, "runtime_freeze", None) is not None:
        protected_paths.add(arguments.runtime_freeze)
    for input_name in ("predictions", "paper_map"):
        if hasattr(arguments, input_name):
            protected_paths.add(getattr(arguments, input_name))
    if arguments.command not in {"validate", "verify-pdfs"}:
        assert_writable_output(arguments.output, protected_paths)
    if arguments.command == "run":
        if arguments.embedding_cache.suffix.lower() != ".npz":
            parser.error("--embedding-cache must name a separate .npz cache file")
        assert_writable_output(arguments.embedding_cache, protected_paths | {arguments.output})
    if arguments.command == "validate":
        payload = validate_dataset(config, manifest, questions, rubric)
    elif arguments.command == "verify-pdfs":
        payload = verify_source_pdfs(manifest, arguments.pdf_dir, arguments.split)
    elif arguments.command == "freeze-run":
        payload = runtime_freeze(config, manifest)
        payload["source_snapshot"] = preserve_runtime_sources(payload, arguments.output)
    elif arguments.command == "template":
        payload = prediction_template(config, questions)
    elif arguments.command == "smoke":
        payload = evaluate(config, manifest, questions, smoke_predictions(config, manifest, questions))
    elif arguments.command == "run-api":
        if arguments.timeout <= 0:
            parser.error("--timeout must be positive")
        freeze_record = None
        if manifest.get("purpose") == "real_paper_text_evaluation" and arguments.split in {"holdout", "all"}:
            freeze_record = validate_runtime_freeze(config, manifest, arguments.runtime_freeze)
        selected_questions = [row for row in questions if arguments.split == "all" or row["split"] == arguments.split]
        payload = run_api(config, selected_questions, arguments.base_url, read_json(arguments.paper_map), arguments.pipeline_revision, arguments.timeout)
        payload["execution"].update({"split": arguments.split, "runtime_freeze": freeze_record})
    elif arguments.command == "run":
        payload = run_pipeline(config, manifest, questions, split=arguments.split, offline=arguments.offline,
                               revision=arguments.pipeline_revision, cache_path=arguments.embedding_cache,
                               device=arguments.device, freeze_path=arguments.runtime_freeze)
    else:
        payload = evaluate(config, manifest, questions, read_json(arguments.predictions))
        payload["provenance"].update({"predictions_sha256": digest(arguments.predictions),
                                      "config_sha256": digest(arguments.config)})
    if arguments.command not in {"validate", "verify-pdfs"}:
        save(arguments.output, payload, protected_paths)
    print(json.dumps(payload.get("metrics", payload) if arguments.command in {"smoke", "validate", "verify-pdfs"} else
                     {"saved": str(arguments.output), "execution_kind": payload.get("execution", {}).get("kind"),
                      "real_paper_holdout_accuracy_claim": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
