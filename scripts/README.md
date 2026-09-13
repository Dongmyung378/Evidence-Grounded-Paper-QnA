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
python -B scripts/validate_error_recovery_ui.py
```

`scripts/evaluate_day36.py`는 업로드와 분석 화면을 검증하고, `scripts/evaluate_evidence_ui.py`는 질문부터 답변과 원문 근거 카드까지 검증한다. `scripts/evaluate_error_recovery_ui.py`는 크기 초과, 잘못된 PDF, 파싱 실패 후 정상 재시도를 검증한다. 평가기는 실제 Streamlit 서버와 Chromium 브라우저를 실행한다. 다시 실행하려면 Node.js, Playwright, 로컬 Edge 또는 Chrome이 필요하다.

Docker Compose 로컬 데모는 다음 명령으로 실제 이미지를 빌드하고 검증한다.

```bash
python -B scripts/evaluate_container.py
python -B scripts/validate_container.py
```

평가기는 사용하지 않는 포트에서 백엔드와 UI를 실행하고 상태 확인, HTTP 응답, 내부 서비스 연결, 비루트 사용자를 검사한 뒤 임시 컨테이너와 테스트 볼륨을 제거한다.

논문 3편과 한영 질문 10개의 실제 컨테이너 질의응답, 오프라인 캐시 재시작, 답변 의미 품질은 다음 명령으로 재현하고 검증한다.

```bash
python -B scripts/run_container_qna_benchmark.py
python -B scripts/build_container_qna_review.py
python -B scripts/validate_container_qna.py
```

전체 벤치마크는 전용 Compose 프로젝트의 임시 실행 볼륨만 제거하고 모델 캐시는 반복 실행을 위해 보존한다. 검증 스크립트만 실행할 때는 Docker가 실행 중일 필요가 없다.

로컬 포트폴리오 제공 범위와 공개 문서의 일관성은 다음 명령으로 확인한다.

```bash
python -B scripts/validate_portfolio_scope.py
```

이 검증은 Docker Compose가 최종 제공 환경인지, 공개 서버와 공개 URL이 완료 조건에서 제외됐는지, 로컬 일지가 Git 추적 대상에서 빠졌는지 확인한다.

현재 답변 정책은 `config/runtime_qna.json`, `config/answer_generation.json`, `config/translation.json`에 있다. 영어 추출형 답변과 한국어 번역의 고정 20문항 평가는 다음 명령으로 재현하고 검증한다.

선택 실행용 답변 확장은 기본 API와 분리한다. `python -B scripts/review_answer_coverage.py`는 저장된 60문항, 원문 연결과 고정 20문항 판정을 모델 실행 없이 검증한다. [실험 검토](../docs/reviews/answer_coverage_KO.md)를 참고한다.

```bash
python -B scripts/evaluate_grounded_generation.py
python -B scripts/build_grounded_generation_review.py
python -B scripts/validate_grounded_generation.py
```

기존 Qwen 실행 성능과 실제 전체 HTTP 흐름 기록은 `data/evaluation/day35_integration_results.json`에 과거 기준선으로 유지한다.

최종 모델 리비전, 시드, 평가 입력과 성능표는 다음 명령으로 다시 만들고 검증한다.

```bash
python -B scripts/evaluate_frozen_retrieval.py
python -B scripts/build_evaluation_freeze.py
python -B scripts/validate_evaluation_freeze.py
```

첫 명령은 고정 리비전의 임베딩과 재정렬 모델을 CPU에서 실행한다. 전용 임베딩 캐시가 없으면 최초 실행 시간이 길어질 수 있다. 두 번째 명령은 현재 검토가 끝난 출처만 해시로 묶으며, 전체 검증에서는 세 번째 명령만 실행해 동결 파일이 임의로 다시 생성되지 않게 한다.
