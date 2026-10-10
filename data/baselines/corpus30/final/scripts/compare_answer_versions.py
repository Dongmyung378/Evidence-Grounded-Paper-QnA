"""Compare historical and current answer construction on saved development evidence."""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

from grounded_generation import generate_evidence_first_answer, load_answer_generation_config
from local_translation import LocalEnglishKoreanTranslator
from model_lock import locked_model
from reranker import load_reranker
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout
from quality_benchmark import IMPLEMENTATIONS


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-ids", nargs="+", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs_path = PROJECT_ROOT / "data/evaluation/answer_evidence_inputs.json"
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    by_id = {row["question_id"]: row for row in inputs["results"]}
    if len(set(args.question_ids)) != len(args.question_ids) or any(q not in by_id for q in args.question_ids):
        parser.error("Question IDs must be unique and present in saved development evidence")
    protected = (PROJECT_ROOT / "data/evaluation").resolve()
    output = args.output.resolve()
    if output.is_relative_to(protected) or output == inputs_path.resolve():
        parser.error("Write new comparisons outside frozen data/evaluation")
    baseline_dir = PROJECT_ROOT / "data/baselines/2026-09-14"
    baseline_module = baseline_dir / "scripts/grounded_generation.py"
    spec = importlib.util.spec_from_file_location("historical_grounded_generation", baseline_module)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    baseline_config = historical.load_answer_generation_config(baseline_dir / "config/answer_generation.json")
    current_config = load_answer_generation_config()
    measured_files = [Path(__file__), inputs_path, baseline_module, baseline_dir/'config/answer_generation.json',
        PROJECT_ROOT/'scripts/grounded_generation.py', PROJECT_ROOT/'scripts/evidence_selector.py',
        PROJECT_ROOT/'scripts/answerability_policy.py', PROJECT_ROOT/'config/answer_generation.json',
        PROJECT_ROOT/'config/answerability_policy.json', PROJECT_ROOT/'scripts/local_translation.py',
        PROJECT_ROOT/'config/translation.json', PROJECT_ROOT/'config/translation_glossary.json',
        PROJECT_ROOT/'config/model_lock.json', PROJECT_ROOT/'config/answer_output_schema.json',
        PROJECT_ROOT/'config/generation_runtime.json',
        *(PROJECT_ROOT/name for name in IMPLEMENTATIONS)]
    measured_hashes = {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in measured_files}
    lock = locked_model("reranker")
    selector = load_reranker(lock["name"], revision=lock["revision"], local_files_only=args.offline, device="cpu")
    translator = None
    rows = []
    for question_id in args.question_ids:
        row = by_id[question_id]
        language = row["question_language"]
        if language == "ko" and translator is None:
            translator = LocalEnglishKoreanTranslator(local_files_only=args.offline)
        compared = {}
        for name, generate, config in (("baseline", historical.generate_evidence_first_answer, baseline_config),
                                       ("current", generate_evidence_first_answer, current_config)):
            started = time.perf_counter()
            result = generate(translator, selector, row["query"], row["retrieved_evidence"], language, config)
            compared[name] = {"response": result["response"], "selected_sentences": result["selected_sentences"],
                              "fallback_used": result["fallback_used"], "wall_seconds": round(time.perf_counter()-started,6),
                              "stage_timings": result.get("stage_timings"),
                              "answerability": result.get("answerability"),
                              "attempts": result.get("attempts", [])}
        rows.append({"question_id": question_id,"paper_id":row["paper_id"],"question":row["query"],"language":language,**compared})
        print(f"Compared {question_id}",flush=True)
    if any(hashlib.sha256(path.read_bytes()).hexdigest()!=value for path,value in measured_hashes.items()):
        raise RuntimeError('Source, settings or input changed during comparison; repeat on a stable revision')
    report={"schema_version":1,"scope":"saved_development_evidence_answer_stage_only",
            "independent_real_paper_holdout":False,"human_review_completed":False,
            "retrieval_repeated":False,"execution_mode":"real_models",
            "comparison_boundary":"Answer selectors/builders are compared using one shared current translation adapter; this is not a full historical-runtime or controlled latency comparison.",
            "comparison_runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "measured_source_sha256":{path.relative_to(PROJECT_ROOT).as_posix():value
                                      for path,value in measured_hashes.items()},
            "inputs_sha256":hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
            "baseline_implementation_sha256":hashlib.sha256(baseline_module.read_bytes()).hexdigest(),
            "current_implementation_sha256":hashlib.sha256((PROJECT_ROOT/'scripts/grounded_generation.py').read_bytes()).hexdigest(),
            "baseline_config_sha256":hashlib.sha256((baseline_dir/'config/answer_generation.json').read_bytes()).hexdigest(),
            "current_config_sha256":hashlib.sha256((PROJECT_ROOT/'config/answer_generation.json').read_bytes()).hexdigest(),
            "current_answerability_policy_sha256":hashlib.sha256((PROJECT_ROOT/'scripts/answerability_policy.py').read_bytes()).hexdigest(),
            "current_answerability_config_sha256":hashlib.sha256((PROJECT_ROOT/'config/answerability_policy.json').read_bytes()).hexdigest(),
            "current_selector_sha256":hashlib.sha256((PROJECT_ROOT/'scripts/evidence_selector.py').read_bytes()).hexdigest(),
            "current_translation_sha256":hashlib.sha256((PROJECT_ROOT/'scripts/local_translation.py').read_bytes()).hexdigest(),
            "current_translation_config_sha256":hashlib.sha256((PROJECT_ROOT/'config/translation.json').read_bytes()).hexdigest(),
            "current_translation_glossary_sha256":hashlib.sha256((PROJECT_ROOT/'config/translation_glossary.json').read_bytes()).hexdigest(),
            "reranker":lock,"translator":translator.metadata() if translator else None,"results":rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"Saved development comparison: {output}")


if __name__ == "__main__":
    main()
