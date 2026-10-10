# External reproduction and answer review

[English](third_party_review.md) | [한국어](third_party_review_KO.md)

This procedure is for another person to verify installation and answers.
Developer-controlled runs in a clean environment and automated agent reviews
are useful checks, but do not establish independent human reproduction.

Install the public test dependencies and follow the README for Python or Docker
setup. The first model-backed request downloads missing pinned revisions and
initializes models. Use offline flags only after the exact models are cached.
The NLLB translation model is restricted to the documented noncommercial scope.

```bash
python -m pip install -r requirements-test.txt
python -B scripts/verify_project.py
docker compose up --build -d
python -B scripts/check_deployment.py --base-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501 --real-cases data/quality_benchmark/deployment/real_pdf_cases.json --output outputs/reviewer-deployment.json
```

Run the client from the repository root on the host while Compose is running.
It uses five authored-PDF cases and two questions on the bundled original PDF.
The UI check covers its health endpoint; browser interaction needs a separate review.

Review actual answers alongside original pages. Record question relevance,
missing list items, comparison targets, quantities, units, conditions,
negation, scope, terminology and unsupported causal claims. Verify that each
citation belongs to the requested paper and supports the answer. Unsupported
questions must return the complete refusal contract with no citations.
Response shape and lexical marker coverage alone do not establish correctness.

Keep the review date, source and configuration hashes, OS, package versions,
device, commands and logs. For every reviewed question, record the question ID,
source location, answer, factual agreement, completeness, citation suitability
and a failure explanation where needed. Identify automated reviews explicitly.
Disclose exclusions instead of extending a sample's result to the whole corpus.

Latency comparisons require the same paper, question, revisions, device and
settings. Separate cold activation from warm inference, and distinguish exact
sentence cache reuse from requests with cleared translation caches. Mark human
external verification complete only after an actual reviewer's run is recorded.
