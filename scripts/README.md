# 스크립트 안내

스크립트는 실행 목적에 따라 다음처럼 구분된다. 기존 재현 경로를 깨지 않기 위해 파일을 과도하게 하위 폴더로 이동하지 않고 이름 규칙과 이 문서로 탐색 구조를 단순화했다.

| 이름 규칙 | 역할 |
|---|---|
| `ingest_*`, `run_ingestion_*` | PDF 파싱과 청크 생성 |
| `search_*` | 단일 질문 검색 실행 |
| `evaluate_*` | 모델과 로드맵 단계 평가 및 결과 저장 |
| `validate_*` | 저장된 결과와 현재 코드의 완료 조건 검증 |
| `test_*` | 독립 실행형 회귀 테스트 |
| `build_*` | 평가 데이터와 보고서 생성 |
| `run_*` | 특정 단계의 실행 진입점 |

전체 상태는 다음 명령으로 확인할 수 있다.

```bash
python -B scripts/verify_project.py
```

실제 PDF부터 답변까지 다시 실행하려면 다음 명령을 사용한다.

```bash
python -B scripts/evaluate_day35.py
python -B scripts/validate_day35.py
```

Streamlit 업로드 및 근거 화면의 저장된 브라우저 수용 결과는 다음 명령으로 검증한다.

```bash
python -B scripts/validate_day36.py
python -B scripts/validate_evidence_ui.py
```

`scripts/evaluate_day36.py`는 업로드와 분석 화면을 검증하고, `scripts/evaluate_evidence_ui.py`는 질문부터 답변과 원문 근거 카드까지 검증한다. 두 평가기는 실제 FastAPI와 Streamlit 서버, Chromium 브라우저를 실행한다. 다시 실행하려면 Node.js, Playwright, 로컬 Edge 또는 Chrome이 필요하다.

실행 성능 정책은 `config/runtime_qna.json`과 `config/generation_runtime.json`에 있다. 실제 전체 흐름의 모델 준비 장치, 검색 및 생성 시간, 시도 횟수는 `data/evaluation/day35_integration_results.json`에서 확인할 수 있다.
