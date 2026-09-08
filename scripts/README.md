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
