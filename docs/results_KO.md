# 평가 결과와 확인된 한계

[English](results.md) | [한국어](results_KO.md)

## 논문 30편의 비교

기존 버전의 개발 실행 전에 평가 입력을 고정했습니다. 이미 레이블을 사용한 10편은 개발 자료이며, 나머지 20편은 과거에 수집 및 검색 확인에 사용했습니다. 이 사용 이력은 운영자 기록이며 독립적인 감사 결과가 아닙니다. 모델의 사전학습 노출은 알 수 없습니다. 답변 경로에는 채점용 레이블을 전달하지 않고 질문과 논문 ID만 전달합니다.

개발 비교는 답변 가능 40개, 근거 부족 10개, 잘못된 전제 10개 그룹입니다.

| 검사 항목 | 기존 | 최종 |
| --- | ---: | ---: |
| 출력 형식 준수 | 120/120 | 120/120 |
| 답변 가능한 질문에 응답 | 23/40 | 23/40 |
| 필수 정보의 어휘 표지 모두 포함 | 8/40 | 8/40 |
| 정답 근거 위치 중 하나를 인용 | 12/40 | 12/40 |
| 근거가 없는 요청을 거절 | 10/10 | 10/10 |
| 잘못된 전제의 요청을 거절 | 0/10 | 0/10 |
| 의미상 정확도 | 독립 측정 없음 | 독립 측정 없음 |

최종 논문 20편의 비교는 답변 가능 80개, 근거 부족 20개, 잘못된 전제 20개 그룹입니다.

| 검사 항목 | 기존 | 최종 |
| --- | ---: | ---: |
| 출력 형식 준수 | 240/240 | 240/240 |
| 답변 가능한 질문에 응답 | 50/80 | 49/80 |
| 필수 정보의 어휘 표지 모두 포함 | 11/80 | 11/80 |
| 정답 근거 위치 중 하나를 인용 | 32/80 | 31/80 |
| 근거가 없는 요청을 거절 | 18/20 | 18/20 |
| 잘못된 전제의 요청을 거절 | 2/20 | 2/20 |
| 의미상 정확도 | 독립 측정 없음 | 독립 측정 없음 |

개발 검색 및 선택 안 세 개는 사전에 고정한 채택 기준을 모두 통과하지 못했습니다. 최종 기본 설정은 기존 검색 및 문장 선택 정책에 원문 대응 기능과 문장 내 스칼라 측정값 인식 수정을 적용합니다. 인접 문맥 복원, 요청 항목 우선 선택과 제한된 재검색은 비활성화된 실험 코드로 보존했습니다. 이번 실험의 중단 판단이며 가능한 최대 성능에 도달했다는 뜻은 아닙니다.

이전에 공개한 CNN 회귀 사례에서 영어 답변은 상관계수 0.86, 감쇠 값 0.01과 미수렴 조건을 이제 보존합니다. 한국어 답변은 여전히 이 사실을 놓칩니다. [원문과 출력](../data/benchmarks/corpus30/results/scalar-regression.json) 및 [작성자 보조 검토](../data/benchmarks/corpus30/results/scalar-regression-review.json)에 이 제한적인 개선을 전체 평가와 구분해 기록했습니다. 두 언어 모두 성공한 사례나 일반적인 정확도 개선으로 해석할 수 없습니다.

확장한 최종 비교에서 전체 품질 향상은 확인되지 않았습니다. 두 언어의 필수 어휘 포함은 11/80으로 같고 응답 유지는 50/80에서 49/80으로 바뀌었습니다. 응답을 잃은 그룹은 `paper-019-slope`입니다. 최종 한국어 번역이 새로 선택한 스칼라 측정값의 그리스 문자를 누락해 수식 검증에서 거절됐으며, 기존 한국어 답변에도 요청한 기울기 추정값은 없었습니다. 응답 상태의 차이와 원문에 따른 운영자 검토이며 독립적인 의미상 품질 판정은 아닙니다. 이 평가 결과로 최종 정책을 조정하지 않았습니다.

언어별 필수 어휘 포함은 영어 43/80, 한국어 15/80이며 두 언어가 같은 그룹에서 모두 충족한 경우는 11/80입니다. 동의어와 전문 용어 표기의 차이가 언어 간 해석을 제한합니다. 전체 출력 360개에 노출된 원문 대응 단위 708개를 저장된 근거에서 재구성할 수 있습니다. [별도 Linux 재현](../data/benchmarks/corpus30/results/reproduction/README_KO.md)은 전체 코퍼스 임베딩을 새로 생성한 뒤 사전 선정한 답변과 원문 대응 및 인용 출력 12개를 재현했습니다. 시간과 재정렬 점수의 부동소수점 차이는 일치 비교에서 제외합니다.

근거: [기존 개발 보고서](../data/benchmarks/corpus30/results/baseline-dev-report.json), [최종 개발 보고서](../data/benchmarks/corpus30/results/final-dev-report.json), [기존 평가 보고서](../data/benchmarks/corpus30/results/baseline-holdout-report.json), [최종 평가 보고서](../data/benchmarks/corpus30/results/final-holdout-report.json). [원시 출력](../data/benchmarks/corpus30/results/)에는 답변과 원문 근거, 진단 정보 및 실행시간이 있습니다. [개발 판단 기록](../data/benchmarks/corpus30/results/comparison.json)에 채택하지 않은 안과 중단 기준을 보관합니다.

그룹 단위 검사는 두 언어가 모두 통과해야 인정합니다. 어휘 표지는 잘못된 문장에 등장해도 통과할 수 있고 올바른 동의어를 놓칠 수 있습니다. 근거 페이지 목록은 모두 인용해야 하는 목록이 아니라 정답이 있는 대체 위치입니다. 거절은 인용이 없는 정해진 근거 부족 응답을 뜻하므로, 잘못된 전제를 정정하는 충분 응답도 거절 점수에는 포함되지 않습니다. 의미상 정확도를 독립적으로 측정한 것은 아닙니다.

기존 버전의 초기 평가 작업은 출력 3개를 생성한 뒤 최종 후보 확정 전에 종료했습니다. 이후 후보 실행은 이전에 공개된 CNN 원문에서 스칼라 인식 오류를 확인해 출력 169개를 생성한 시점에 중단했습니다. 수정에는 그 알려진 원문을 사용했으며 corpus30의 출력은 사용하지 않았습니다. 운영자 기록상 두 중단 실행의 답변과 점수는 열람하거나 점수 기반 개선에 사용하지 않았습니다. 진행 건수와 사례 ID는 확인했습니다. 최종 소스를 고정한 뒤 정식 전체 비교를 수행했습니다. 실행 순서의 이탈을 공개하며 비사용 사실을 독립적으로 증명하는 것은 아닙니다. 공개된 평가 질문은 앞으로 회귀 자료로 취급해야 합니다.

정식 품질 검증은 같은 CPU 호스트의 독립된 두 프로세스에서 겹쳐 실행했습니다. 각각 Torch 스레드 8개와 별도의 임베딩 캐시를 사용합니다. 질문당 실행시간은 진단용으로 보존하며 통제된 지연시간 비교로 해석할 수 없습니다. [실행 조건](../data/benchmarks/corpus30/results/execution_context.json)에 기록했습니다. 세 개발 전략은 각각 따로 실행했으며, 이 실행시간도 최종 서비스의 부하 성능을 뜻하지 않습니다.

추론 의존성과 지정된 모델 캐시를 준비한 뒤 저장소 최상위 폴더에서 재실행합니다.

```bash
python -B scripts/validate_corpus_benchmark.py
python -B scripts/replay_corpus_benchmark.py --variant final --split holdout --offline --output outputs/final-replay.json
python -B scripts/replay_corpus_benchmark.py --variant baseline --split holdout --offline --output outputs/baseline-replay.json
python -B scripts/corpus_benchmark.py evaluate --predictions outputs/final-replay.json --output outputs/final-replay-report.json
```

재실행 명령은 [측정 당시 소스](../data/baselines/corpus30/)를 별도의 임시 폴더에 복원하고 밀집 임베딩을 새로 생성합니다. 공개 검증은 모델이나 비공개 PDF 없이 기록된 지표를 다시 계산합니다. PDF 파싱과 시각적 추론은 답변 벤치마크의 범위 밖이며, API 및 업로드 테스트에서 인터페이스를 별도로 확인합니다. 구현은 [요청 항목 추적과 원문 대응](evidence_planning_KO.md)에 설명했습니다.

## 이전 논문 4편의 평가와 재현

아래 수치는 보존한 이전 구현의 결과입니다. 별도의 과거 회귀 근거이며 현재 확장 평가의 결과가 아닙니다.

### 이전 실논문 평가

각 분할에는 논문 2편, 이중 언어 의미 그룹 16개와 출력 32개가 있습니다. 그룹은 답변 가능 10개, 근거 부족 4개와 잘못된 전제 2개로 구성됩니다. 두 언어가 모두 조건을 만족해야 그룹 단위 통과로 인정합니다. 개발 자료는 구현 개선에 사용했으며 평가 논문은 최종 구현을 고정한 뒤 실행했습니다. 결과가 공개된 현재는 향후 개선에서 이 평가 자료를 회귀 검증 근거로 취급해야 합니다.

| 자동 검사 | 개발 자료 | 고정 구현의 평가 논문 |
| --- | ---: | ---: |
| 출력 형식 준수 | 출력 32/32개 | 출력 32/32개 |
| 답변 가능한 요청에 응답 | 그룹 8/10개 | 그룹 7/10개 |
| 근거가 부족한 요청을 거절 | 그룹 4/4개 | 그룹 1/4개 |
| 잘못된 전제의 요청을 거절 | 그룹 2/2개 | 그룹 0/2개 |
| 부정 사례에 답변을 반환 | 그룹 0/6개 | 그룹 5/6개 |
| 필요한 페이지를 모두 인용 | 답변 가능 그룹 7/10개 | 답변 가능 그룹 5/10개 |
| 필요한 어휘 표지를 모두 포함 | 답변 가능 그룹 4/10개 | 답변 가능 그룹 1/10개 |
| 의미상 정확도 | 독립적으로 측정하지 않음 | 독립적으로 측정하지 않음 |

근거: [개발 보고서](../data/quality_benchmark/runs/2026-10-10-final/real-dev-completion-report.json), [평가 보고서](../data/quality_benchmark/runs/2026-10-10-final/real-holdout-report.json), [개발 원출력](../data/quality_benchmark/runs/2026-10-10-final/real-dev-completion-predictions.json), [평가 원출력](../data/quality_benchmark/runs/2026-10-10-final/real-holdout-predictions.json).

평가 논문에서 답변 가능한 요청에 대한 응답은 영어 10/10개와 한국어 7/10개였습니다. 근거 부족 요청의 거절은 각각 3/4개와 2/4개였으며, 잘못된 전제의 거절은 1/2개와 0/2개였습니다. 응답 상태와 어휘 및 페이지 포함 여부는 사실 정확성과 구별해야 합니다. 논문에 있는 사실도 질문과 무관할 수 있고, 기대한 페이지를 인용하면서 요청한 내용을 누락할 수도 있습니다.

[동결 기록](../data/quality_benchmark/runs/2026-10-10-final/real-paper-runtime-freeze.json)과 [측정 당시 구현 원본](../data/quality_benchmark/runs/2026-10-10-final/real-paper-runtime-freeze-sources/)에는 구현 파일 20개와 설정 10개를 보존했습니다. 평가 전후에 해시를 검사합니다. 이전 후보 실행은 주 비교에서 제외했으며, 운영자 기록에 따르면 해당 출력을 이후 개발 개선에 사용하지 않았습니다. 이 비사용 사실은 독립적으로 확인된 주장이 아닙니다. 새로운 정확도 추정에는 새로운 평가 논문과 독립적인 사실 검토가 필요합니다.

### 원문 검토에서 발견한 실패

다음 내용은 에이전트가 저장된 답변과 원문을 살펴본 결과이며, 독립적으로 채점한 의미상 정확도는 아닙니다.

| 영역 | 발견한 실패 | 평가 시 확인할 점 |
| --- | --- | --- |
| 수치 결과 | 인용 청크에 CNN 상관계수 R = 0.86, 감쇠값 0.01과 미수렴 결과가 있지만 답변에서 모두 빠졌습니다. | 올바른 페이지 검색만으로 답변 완전성을 판단할 수 없습니다. |
| 목록 | 한국어 증강 답변이 수평 뒤집기를 누락하고, 기준 모델 목록의 항목이 청크 경계에서 사라질 수 있습니다. | 요청한 항목과 조건을 하나씩 확인해야 합니다. |
| 비교 | 재학습 답변에서 근사 방식과 비용의 비교가 빠지고, 연산자 답변에서 고정 정책 조건이 누락됩니다. | 비교 대상과 성립 범위를 함께 보존해야 합니다. |
| 전문 용어 번역 | Hessian-vector products가 “제품”으로 번역되고 거리와 미분 가능성 관련 용어가 오역됩니다. | 용어 보호와 수치 검증만으로 의미 보존을 입증할 수 없습니다. |
| 근거 부족 요청 | 임상시험 질문에 Atari 결과를 답하고, 금융 및 신용 관련 요청에 무관한 최적화 사실을 답합니다. | 유효한 출처만으로 질문 관련성을 판단할 수 없습니다. |
| 잘못된 전제 | 보편적인 수축 성질을 부정하면서도 요청한 보편적 수축률 없이 충분한 답변으로 표시합니다. | 부정 사례의 응답이 항상 긍정적 사실의 날조를 뜻하지는 않으며 거절 상태도 검토해야 합니다. |

기본 실행 경로는 추출과 번역을 유지합니다. 생성 모델의 도입 판단은 [범위를 제한한 비교 실험](experiments_KO.md)에 별도로 기록했습니다.

### 실행시간 측정

Windows CPU 측정은 같은 [보정 논문의 청크 49개](../data/quality_benchmark/runs/2026-10-10-final/runtime-calibration-chunks.jsonl)를 사용했습니다. 준비 실행 한 번을 제외하고 같은 온도 질문을 10회 반복했으며 Torch 스레드는 8개입니다. 첫 실행 시간에는 초기 모델 활성화와 지연된 한국어 모델 로딩이 포함됩니다.

| 조건 | 첫 실행 | 반복 실행 p50 | 반복 실행 p95 | 최대 프로세스 RSS |
| --- | ---: | ---: | ---: | ---: |
| [영어](../data/quality_benchmark/runs/2026-10-10-final/runtime-en-enabled.json) | 19.912초 | 2.134초 | 2.201초 | 1.31 GiB |
| [한국어, 같은 문장 캐시](../data/quality_benchmark/runs/2026-10-10-final/runtime-ko-enabled.json) | 25.861초 | 2.301초 | 2.379초 | 4.69 GiB |
| [한국어, 요청마다 문장 캐시 초기화](../data/quality_benchmark/runs/2026-10-10-final/runtime-ko-disabled.json) | 26.756초 | 4.337초 | 4.730초 | 4.69 GiB |

이는 동일 질문 반복 측정이며 다양한 질문의 지연시간, 처리량, 동시성이나 SLA를 측정한 결과가 아닙니다. 문장 출력 캐시를 비워도 로딩한 모델은 유지하며, 새로운 프로세스도 운영체제 파일 캐시까지 비우지는 않습니다. 중첩된 단계별 시간을 중복해서 더하면 안 됩니다. 측정 환경은 Python 3.12.7, CPU Torch 2.14.1, Transformers 5.16.1, Sentence Transformers 6.0.0과 NumPy 1.26.4입니다.

### 환경 및 HTTP 재현

Python 3.12.15와 CPU Torch 2.5.1을 설치한 새로운 Linux 컨테이너에서 기존 평가 답변 32개와 인용 내용 및 보고서 지표를 모두 재현했습니다. 밀집 검색 임베딩은 다시 생성했으며, 재정렬 점수의 최대 차이는 8.94e-6이었습니다. 작성한 PDF 5개 사례와 원본 논문의 영어 및 한국어 질문 2개를 포함한 HTTP 사례 7개가 통과했습니다. 원본 논문은 텍스트 페이지 11개와 청크 63개로 분석됐고 UI 상태 확인도 통과했습니다.

[환경 요약](../data/quality_benchmark/reproduction/separate_environment_summary.json), [답변 비교](../data/quality_benchmark/reproduction/holdout-comparison.json), [HTTP 비교](../data/quality_benchmark/reproduction/api-comparison.json)에서 확인할 수 있습니다. 당시 테스트 367개와 공개 검증기 9개가 통과했습니다. 이는 측정한 구현의 결과이며 이후 모든 코드 변경의 결과를 뜻하지 않습니다.

Linux 환경은 WSL2를 통해 같은 물리 장비를 사용했으며 고정된 모델 파일은 읽기 전용 캐시로 공유했습니다. 최초 모델 다운로드와 브라우저 직접 조작은 검증하지 않았고 비공개 코퍼스 검사는 건너뛰었습니다. 에이전트가 수행한 실행으로 독립적인 사람의 평가를 입증하지는 않습니다. 장비별 실행 로그는 로컬에 보관하고 공개 요약과 비교 자료에는 측정 결론을 유지합니다.

### 재현 방법

공개 산출물의 무결성과 회귀 검증을 실행합니다.

```bash
python -m pip install -r requirements-test.txt
python -B scripts/verify_project.py
```

실제 모델 실행에는 `requirements.txt`가 필요합니다. `--offline`은 고정된 모델이 모두 캐시에 있어야 하며 최초 다운로드 시에는 생략합니다. 공개 산출물을 덮어쓰지 않고 개발 출력과 보고서를 생성합니다.

```bash
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json validate
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json run --split dev --offline --device cpu --embedding-cache outputs/real-dev-embeddings.npz --output outputs/real-dev-predictions.json
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json evaluate --predictions outputs/real-dev-predictions.json --output outputs/real-dev-report.json
```

이미 공개한 평가 자료는 [스크립트 안내](../scripts/README_KO.md)의 동결 및 실행 명령으로 회귀 재생할 수 있습니다. 재생은 새로운 미공개 평가 결과를 제공하지 않습니다. 분할 및 채점 규칙은 [품질 평가](quality_benchmark_KO.md)를 참고합니다.

업로드와 HTTP 경로는 Compose를 실행한 뒤 호스트에서 검사합니다.

```bash
python -B scripts/check_deployment.py --base-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501 --real-cases data/quality_benchmark/deployment/real_pdf_cases.json --output outputs/deployment/reproduction.json
```

[이용 허가된 PDF와 사례](../data/quality_benchmark/deployment/README.md)는 공개 소스에 포함했습니다. 다른 사람이 사실을 검토할 때는 [외부 검증 절차](third_party_review_KO.md)를 사용합니다. 과거 검색 수치와 답변 검토는 당시 구현 원본에 연결해 [과거 평가](evaluation_KO.md)에 보존했습니다.
