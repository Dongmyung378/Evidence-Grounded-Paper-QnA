# 최종 답변 품질 평가

[English](quality_benchmark.md) | [한국어](quality_benchmark_KO.md)

기존 전체 논문의 평가는 [30편 확장 벤치마크](../data/benchmarks/corpus30/README_KO.md)에서 수행합니다. 영어와 한국어를 묶은 180개 의미 그룹을 평가합니다. 아래 평가 도구의 합성 입력과 별도 논문 4편의 회귀 평가는 각각 분리한 측정입니다.

이 평가 도구는 검색, 답변 거부 정책, 답변 구성, 번역, 검증, 대체 응답을
거친 최종 결과를 평가합니다. 검색 단계에서 거부를 결정했다는 기록만으로
최종 답변 거부에 성공했다고 집계하지 않습니다.

기본 자료는 직접 작성한 가상의 연구 텍스트 4개입니다. 실제 신규 논문
평가셋이 아닙니다. 영문과 한국어 출력 40개는 독립 의미 그룹 20개에
해당하며, 개발용과 보류 평가용에 각각 10개 그룹을 배정했습니다.
이 자료의 점수는 평가 도구와 모델 연결을 확인하는 데 사용합니다.
새로운 실제 논문에 대한 정확도나 제3자 검증 완료를 입증하지 않습니다.

의미 그룹은 번역 질문을 묶는 집계 단위입니다. 가상 텍스트는 통제된
작성 구조를 공유하므로 서로 다른 ID를 통계적으로 독립적인 실험이나
과학 분야 전반에 대한 일반화의 증거로 해석하지 않습니다.

## 모델 다운로드 없이 평가 도구 확인

저장소 루트에서 Python 3.12로 실행합니다.

```bash
python scripts/validate_quality_benchmark.py
python -m unittest discover -s tests -p test_quality_benchmark.py -v
python scripts/quality_benchmark.py smoke --output data/quality_benchmark/smoke_report.local.json
```

`smoke`는 기준 답안을 사용해 의도적으로 정답 출력을 만듭니다.
결과에 `stub_fixture`와 `gold_used_to_construct_outputs: true`를 표시하므로
만점이 나와도 실제 질의응답 성능으로 해석할 수 없습니다.
출력 경로는 직접 지정하며 기존 `data/evaluation`의 동결 자료를 덮어쓰지 않습니다.
동결된 평가 자료, 기준 자료, 설정과 소스 디렉터리, 가상 평가 입력,
현재 평가 중인 출력 파일을 덮어쓰는 경로는 모델 초기화 전에 거부합니다.

## 실제 모델을 사용하는 최종 출력 평가

저장소의 실행 의존성을 설치하고 `config/model_lock.json`에 지정된 모델
리비전을 준비한 뒤 실행합니다. 모델 준비 방법은 저장소 설치 안내를
참고합니다. `--offline`은 누락된 모델 파일을 새로 다운로드하지 않습니다.

```bash
python scripts/quality_benchmark.py run --split dev --offline --output data/quality_benchmark/dev_predictions.local.json
python scripts/quality_benchmark.py evaluate --predictions data/quality_benchmark/dev_predictions.local.json --output data/quality_benchmark/dev_report.local.json
```

`run`은 원문 페이지로 청크를 구성한 뒤 `ProductionRetrieval`과
`GroundedQAPipeline`을 호출합니다. 임베딩 모델, 재순위화 모델, 거부 정책,
답변 구성과 한국어 번역 모델이 실제로 실행됩니다. 정답 페이지, 기준 답안,
질문 유형, 답변 가능 여부는 검색과 답변 생성에 전달하지 않습니다.
PDF 파싱은 거치지 않으므로 문서 업로드와 파싱을 검증한 결과는 아닙니다.
임베딩 캐시는 기존 실험 캐시와 다른 파일을 사용합니다.

검색 모델을 CPU에서 실행하려면 `--device cpu`를 지정합니다.
번역 모델은 `config/translation.json`의 장치 정책을 따릅니다.
의존성이나 모델이 누락되면 초기화 오류가 발생합니다. 개별 질문 실행이
실패해도 해당 질문을 제거하지 않고 유효하지 않은 출력으로 분모에 남깁니다.

개발용 자료로 설정을 조정한 뒤 설정을 고정하고 가상 자료의 보류 평가용을
실행합니다.

```bash
python scripts/quality_benchmark.py run --split holdout --offline --output data/quality_benchmark/holdout_predictions.local.json
python scripts/quality_benchmark.py evaluate --predictions data/quality_benchmark/holdout_predictions.local.json --output data/quality_benchmark/holdout_report.local.json
```

실행 결과에는 설정 파일과 구현 파일의 SHA256, 의존성 버전, 운영체제와
모델 리비전 설정을 기록합니다. 배포 버전에서는
`--pipeline-revision 정확한커밋값`을 지정합니다. 기본값은 변경 중인 로컬
코드를 의미하며 재현 가능한 배포 태그를 대신하지 않습니다.

## 저장된 최종 출력 평가와 API 실행

```bash
python scripts/quality_benchmark.py template --output data/quality_benchmark/predictions_template.local.json
python scripts/quality_benchmark.py evaluate --predictions 실제출력.json --output 평가결과.json
```

출력 양식은 `not_run`으로 표시되며 답변은 비어 있습니다. 각 `output`을
내부 파이프라인 전체 결과나 공개 `/question` 응답으로 채웁니다.
질문 ID를 유지하고 실제 실행 방식을 표시해야 합니다. 선언한 분할의 모든
질문을 한 번씩 포함해야 하며, 누락되거나 중복되거나 알 수 없는 ID는 거부합니다.
실행 실패는 빈 출력으로 남길 수 있지만 답변 거부 성공으로 집계되지 않습니다.

이미 API가 준비되어 있다면 평가 대상 원문을 업로드하고 분석한 뒤,
평가 논문 ID와 API가 반환한 `paper-32자리16진수` ID를 연결한 JSON 파일을
만듭니다. 서로 다른 원문을 같은 실행 논문으로 연결할 수 없습니다.

```bash
python scripts/quality_benchmark.py run-api --base-url http://localhost:8000 --paper-map 논문연결.json --pipeline-revision 정확한커밋값 --output API출력.json
python scripts/quality_benchmark.py evaluate --predictions API출력.json --output API평가.json
```

API에는 실행 논문 ID와 질문만 전달합니다. 기본 가상 자료는 JSON 원문으로
제공되므로 그대로 PDF 업로드할 수 없습니다. 기본 자료의 직접 실행에는
`run`을 사용하고, API 실행은 동일한 원문 문서를 준비했거나 실제 논문
평가셋을 새 버전으로 구성한 경우에 사용합니다.

## 데이터 분리와 집계 기준

- 원문 논문은 하나의 분할에만 속합니다. 논문 ID가 달라도 원문 SHA256이
  같으면 개발용과 보류 평가용에 동시에 포함할 수 없습니다.
- 같은 분할에서도 동일한 원문을 다른 ID로 반복해 표본 수를 부풀릴 수 없습니다.
- 의미 그룹마다 영문과 한국어 질문을 각각 하나씩 포함합니다. 원문, 분할,
  질문 유형, 답변 가능 여부와 정답 페이지가 일치해야 합니다.
- 각 분할은 사실 확인, 수치, 목록, 비교, 한계의 다섯 유형과 답변 가능,
  근거 없음, 잘못된 전제의 세 종류를 모두 포함합니다.
- 주요 지표는 의미 그룹을 기준으로 계산합니다. 두 언어가 모두 통과해야
  그룹의 답변 유지나 거부 성공으로 집계합니다. 언어별 지표는 따로 제시하며
  번역 질문을 독립 표본으로 두 번 세지 않습니다.
- 거부 성공에는 최종 `insufficient`, 정해진 거부 문구, 빈 인용 목록,
  거부 사유가 모두 필요합니다. 임의의 내용을 답하면서 거부 라벨만 붙이거나
  거부 문구에 답변 가능 라벨을 붙인 결과는 유효하지 않습니다.
- 답변 유지율은 유효한 형식으로 실제 답변을 반환한 비율입니다. 과학적
  내용의 정확도를 의미하지 않습니다. 오거부, 근거 없는 답변, 출력 오류와
  검증 실패에 따른 대체 응답은 별도로 확인할 수 있습니다.
- 정답 페이지 포함률은 인용 위치만 확인합니다. 페이지 번호가 맞다는
  이유만으로 인용 문장이 주장 전체를 뒷받침한다고 판단하지 않습니다.
- 분모가 0인 지표는 `null`로 표시하며 만점으로 처리하지 않습니다.

`config/quality_benchmark.json`에서 입력 파일을 지정합니다.
`freeze.json`은 원문 목록, 질문, 평가 기준의 정확한 파일 해시를 고정하고,
원문 목록은 각 원문 파일의 해시를 기록합니다. 파일이 바뀌면 평가를
중단합니다. 의도적인 변경은 새 평가 버전으로 검토해야 합니다.
기존 실험 자료는 이 새 평가셋과 분리되어 있습니다.

## 독립적인 실제 논문 평가 준비

`data/quality_benchmark/real_holdout_template.json`은
`not_prepared_not_run` 상태입니다. 기존 자료와 설정 조정에 사용하지 않은
논문을 선정하고 출처와 배포 권한을 확인한 뒤 원문 해시, 질문과 평가 기준을
고정해야 합니다. 같은 논문의 번역 질문은 같은 분할에 유지하며 개발용에서
설정을 확정한 다음 보류 평가용의 최종 출력을 확인합니다.

고정된 평가 기준은 사실 정확성, 완전성, 인용 뒷받침을 사람이 검토하도록
정의합니다. 검토 기록에는 질문 ID, 정확한 출력과 평가 입력 해시, 검토자 ID,
작성자와의 관계, 검토 시각, 각 항목의 `pass`, `partial`, `fail` 판정과 근거를
남겨야 합니다. 원문과 실제 답변을 함께 확인하고 누락된 조건이나 잘못된
주장을 구체적으로 적습니다.

작성자가 만든 가상 자료와 자체 검토를 독립 검토라고 표시할 수 없습니다.
이 도구는 자동 의미 정확도를 `null`, 독립적인 사람 검토를 `false`, 제3자
검증을 `not_attested`로 유지합니다. 해당 검증을 주장하려면 실제 검토자와
판정 근거가 포함된 별도 기록이 필요합니다.
