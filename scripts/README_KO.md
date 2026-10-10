# 스크립트

[English](README.md) | [한국어](README_KO.md)

이 디렉터리에는 현재 사용하는 실행, 평가와 검증 진입점만 남겼습니다. 과거 단계별 스크립트는 포트폴리오 브랜치에서 제거했습니다.

## 주요 명령

저장소 최상단에서 공개 검증용 의존성을 설치하고 저장한 산출물부터 검증합니다.

```bash
python -m pip install -r requirements-test.txt
python -B scripts/verify_project.py
python -B scripts/quality_benchmark.py validate
python -B scripts/quality_benchmark.py smoke --output outputs/quality-smoke.json
python -B scripts/ask_paper.py --help
```

`verify_project.py`는 Git에 포함된 산출물만 있는 공개 저장소 복제본에서도 동작합니다. 비공개 논문 PDF 30편과 생성된 Dense 캐시가 로컬에 있으면 이를 감지해 원본 검증까지 자동으로 실행합니다.

## 직접 준비한 PDF 수집

`YOUR_PDF_DIRECTORY`를 텍스트 기반 영문 PDF가 들어 있는 경로로 바꿉니다. 입력은 `paper-*.pdf` 이름에 해당하는 파일만 선택합니다. 포함된 코퍼스를 보존하도록 별도 출력 경로를 지정합니다.

```bash
python -B scripts/run_ingestion_pipeline.py --papers-dir YOUR_PDF_DIRECTORY --pages-output outputs/ingestion/pages.jsonl --chunks-output outputs/ingestion/chunks.jsonl --report outputs/ingestion/ingestion_report.json
```

## 과거 평가 도구

`data/evaluation` 원본 바이트를 보존하고 현재 검증을 위해 다시 생성하지 않습니다. 아래 파일명은 과거 절차의 참고이며 공개 복제본의 빠른 실행 명령이 아닙니다.

- `evaluate_frozen_retrieval.py`와 `evaluate_abstention.py`는 과거 산출물 경로에 고정해서 쓰며 `--output` 옵션이 없습니다.
- `evaluate_grounded_generation.py`와 `run_container_qna_benchmark.py`는 `--output`을 받지만 기본값은 보존해야 할 과거 산출물 경로입니다.

빌더는 저장한 실행 결과와 검토 라벨을 최종 산출물로 만들고, 검증기는 모델을 내려받지 않고 산출물 일관성을 확인합니다.

## 현재 모델 검증

모델을 직접 실행하려면 전체 실행 의존성을 설치해야 합니다. `--offline`은 고정한 모든 모델 리비전이 캐시에 있을 때만 사용하고 최초 다운로드에서는 제거합니다. 실행시간 측정의 `--local-files-only`도 같은 제한을 적용합니다.

```bash
python -m pip install -r requirements.txt
```

현재 모델 실행과 개발용 비교 명령:

```bash
python -B scripts/quality_benchmark.py run --split dev --offline --output outputs/quality-dev-predictions.json
python -B scripts/quality_benchmark.py evaluate --predictions outputs/quality-dev-predictions.json --output outputs/quality-dev-report.json
python -B scripts/compare_answer_versions.py --question-ids q-041-en q-041-ko q-042-en q-043-ko --offline --output outputs/development-answer-comparison.json
python -B scripts/check_local_demo.py --offline --output outputs/local-demo-check.json
python -B scripts/calibrate_answerability.py --predictions outputs/quality-dev-predictions.json --offline --output outputs/answerability-calibration.json
python -B scripts/benchmark_runtime.py --help
```

개발용 비교는 저장한 검색 근거를 재사용해 답변 구성 단계만 비교합니다. 두 빌더 모두 현재 번역기를 사용하므로 과거 전체 실행의 재현, 조건을 통제한 지연시간 비교 또는 독립 최종 평가의 정확도를 입증하지 않습니다. 실행 시간 보고서는 최초 활성화와 반복 요청의 p50 및 p95를 구분하고 환경 및 설정 해시를 보존합니다. 하위 단계 시간은 상위 단계 시간에 포함되므로 중복해서 합산하지 않습니다.

실제 논문 평가는 개발 자료에서 설정을 확정한 뒤 코드와 설정을 고정하고 보류 자료를 실행합니다.

```bash
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json validate
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json run --split dev --offline --device cpu --embedding-cache outputs/real-dev-embeddings.npz --output outputs/real-dev-predictions.json
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json freeze-run --output outputs/real-runtime-freeze.json
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json run --split holdout --runtime-freeze outputs/real-runtime-freeze.json --offline --device cpu --embedding-cache outputs/real-holdout-embeddings.npz --output outputs/real-holdout-predictions.json
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json evaluate --predictions outputs/real-holdout-predictions.json --output outputs/real-holdout-report.json
```

`check_local_demo.py`는 같은 프로세스의 테스트 클라이언트를 사용합니다. API와 UI를 실행한 상태에서 공개 복제본의 호스트에서 HTTP 검사를 실행합니다.

```bash
python -B scripts/check_deployment.py --base-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501 --real-cases data/quality_benchmark/deployment/real_pdf_cases.json --output outputs/deployment/current-check.json
```

[포함한 PDF와 검사 예시](../data/quality_benchmark/deployment/README.md)는 원본 PDF, 질문과 필요한 어휘 그룹을 제공하며 원본 바이트 해시를 확인합니다. 검사 자료는 Docker 이미지에 복사되지 않습니다. 검사는 PDF 업로드, 파싱과 HTTP 질의응답을 실행하며 UI는 상태 엔드포인트만 확인합니다. 브라우저 화면 조작은 검증하지 않습니다. 클라이언트 해시와 응답만으로 서버 코드와 패키지 버전을 확인할 수 없으므로 별도 기록이 필요합니다. 외부 검토 기록은 [제3자 검토 절차](../docs/third_party_review_KO.md)를 따릅니다.

실행 성능의 `--translation-cache-mode disabled`는 각 요청 전에 번역 결과 캐시만 비우고 모델은 유지합니다. `enabled`와 같은 질문 및 반복 횟수로 비교하세요. 두 모드 모두 동일 질문의 반복 측정이므로 새로운 질문의 처리량으로 해석하지 않습니다.

## 논문 30편 평가와 측정 당시 소스 재실행

```bash
python -B scripts/corpus_benchmark.py validate
python -B scripts/corpus_benchmark.py run --split dev --variant current --offline --output outputs/current-dev.json
python -B scripts/corpus_benchmark.py evaluate --predictions outputs/current-dev.json --output outputs/current-dev-report.json
python -B scripts/validate_corpus_benchmark.py
python -B scripts/replay_corpus_benchmark.py --variant final --split holdout --offline --output outputs/final-holdout-replay.json
```

`corpus_benchmark.py`는 새로운 실행 결과를 `outputs/` 아래에만 쓰고, 완료된 출력의 덮어쓰기를 거부합니다. 재실행 도구는 측정 당시 구현을 별도의 임시 소스 폴더에 복원하고 임베딩을 새로 생성합니다. `--variant baseline`으로 기존 구현도 재현할 수 있습니다. 추론에는 `requirements.txt`와 지정된 모델 캐시가 필요하며, 공개 검증은 `requirements-test.txt`로 수행합니다.

별도의 `experiment_premise_verifier.py`는 정상 전제 대조군을 포함해 채택하지 않은 실험을 재실행합니다. 이 분류기는 기본 서비스의 의존성이나 채택한 전제 정책이 아닙니다. [실험 기록](../data/benchmarks/corpus30/experiments/premise-verifier/README_KO.md)에 판단 근거를 설명했습니다.
