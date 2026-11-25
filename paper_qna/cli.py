"""Command-line interface for Evidence-Grounded Paper Q&A v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_qna.pipeline import PaperQnAEngine
from paper_qna.web_demo import serve_demo


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="paper-qna",
        description="Evidence-Grounded Paper Q&A v2 — Local bilingual scientific paper Q&A, ONNX INT8 NLI gate, and interactive demo.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show local model, ONNX INT8 NLI, Conformal Gate, and LLM server status.")
    p_status.add_argument("--llm-url", default=None, help="Optional local OpenAI-compatible LLM URL (e.g. http://127.0.0.1:8080).")

    p_demo = sub.add_parser("demo", help="Launch the interactive local web UI (FastAPI + Uvicorn).")
    p_demo.add_argument("--host", default="127.0.0.1")
    p_demo.add_argument("--port", type=int, default=8000)
    p_demo.add_argument("--no-browser", action="store_true")
    p_demo.add_argument("--llm-url", default=None)
    p_demo.add_argument("--llm-model", default=None)
    p_demo.add_argument("--neural-retrieval", action="store_true")

    p_ask = sub.add_parser("ask", help="Ask an English or Korean question against a Corpus30 paper or local PDF.")
    p_ask.add_argument("--paper-id", default=None, help="Corpus30 paper ID (e.g. paper-030).")
    p_ask.add_argument("--pdf", default=None, help="Path to a local text-based English PDF.")
    p_ask.add_argument("--question", required=True, help="English or Korean question.")
    p_ask.add_argument("--nli-threshold", type=float, default=0.95)
    p_ask.add_argument("--llm-url", default=None)
    p_ask.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)

    if args.command == "status":
        eng = PaperQnAEngine(llm_url=args.llm_url)
        print(json.dumps(eng.status(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "demo":
        serve_demo(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            use_neural_retrieval=args.neural_retrieval,
        )
        return 0

    if args.command == "ask":
        eng = PaperQnAEngine(llm_url=args.llm_url)
        pid = args.paper_id
        if args.pdf:
            pdf_path = Path(args.pdf)
            reg = eng.register_pdf(pdf_path.read_bytes(), filename=pdf_path.name)
            pid = reg["paper_id"]
        if not pid:
            raise SystemExit("Provide either --paper-id or --pdf.")
        res = eng.ask(pid, args.question, nli_threshold=args.nli_threshold)
        if args.as_json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            resp = res["response"]
            print(f"[{resp['sufficiency'].upper()}] {res['paper']['paper_id']} — {res['paper']['title']}")
            print(f"Answer   : {resp['answer']}")
            if resp.get("abstention_reason"):
                print(f"Reason   : {resp['abstention_reason']}")
            print(f"Sci NLI  : contra={res['nli']['max_contra']:.4f} (thr={res['nli']['threshold']:.2f}, {res['nli'].get('engine','fp32')}) | {res['nli']['latency_ms']:.1f} ms")
            if res.get("conformal"):
                c = res["conformal"]
                print(f"Conformal: P(ans)={c['probability']:.4f} (alpha={c['alpha']}, thr={c['threshold']:.4f})")
            for ev in res.get("cited_evidence", []):
                print(f"  Cited  : [{ev['evidence_id']}] p.{ev['page']} ({ev['section']})")
        return 0

    return 1
