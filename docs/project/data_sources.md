# 데이터 출처와 취급 정책

## 목적

이 문서는 프로젝트에서 사용하는 영어 논문과 평가 데이터의 출처, 역할, 라이선스 확인 방식, 공개 범위를 기록한다.

## arXiv 논문

- 공식 주소: https://arxiv.org/
- 용도: PDF 업로드, 파싱, 청크 생성, 검색, 질문과 답변 검증
- 논문별 식별자와 출처 URL을 `data/metadata/paper_manifest.csv`에 기록한다.
- PDF를 내려받을 수 있다는 사실만으로 재배포 권한이 생기지 않는다.
- 논문별 라이선스를 확인하고 재배포가 명시적으로 허용되지 않으면 원본 PDF는 로컬에만 둔다.

## QASPER

- 데이터셋: https://huggingface.co/datasets/allenai/qasper
- 논문: https://arxiv.org/abs/2105.03011
- 라이선스: CC BY 4.0
- 용도: 과학 논문 질문, 답변, 근거 구조 분석 및 평가 데이터 참고
- 확인된 분할: train 888편, validation 281편, test 416편
- MVP는 텍스트 근거만 사용하며 표, 그림, 수식 근거는 제외한다.
- 로컬 경로: `data/raw/qasper/`

## 현재 논문 코퍼스

| 역할 | 논문 ID | 편수 | 사용 목적 |
|---|---|---:|---|
| 고정 정확도 벤치마크 | `paper-001`부터 `paper-010` | 10 | 검증된 한영 질문 40개 |
| 확장 강건성 세트 | `paper-011`부터 `paper-030` | 20 | 파싱과 모델 실행 점검 |
| 전체 | `paper-001`부터 `paper-030` | 30 | 563페이지, 2,463청크 |

추가 20편은 수동 검증 질문과 Gold 근거를 만들기 전까지 정확도 지표에 포함하지 않는다. 이렇게 하면 기존 40문항 기준선의 비교 가능성을 유지하면서 다양한 PDF와 분야를 시험할 수 있다.

특이 사항은 다음과 같다.

- `paper-020`은 arXiv:2608.26855v1이며 arXiv 비독점 배포 라이선스이므로 원본 PDF를 공개 저장소에 포함하지 않는다.
- `paper-026`은 arXiv:2607.03214v1이며 CC BY-NC-ND 4.0 조건을 따른다.
- 수식 비중이 높은 논문은 파서 강건성에는 유효하지만 수식 해석은 MVP 범위 밖이다.

## 평가 데이터 기준

평가 레코드는 다음 정보를 포함한다.

- `paper_id`
- `question`, `question_language`
- `gold_answer`, `gold_evidence_text`
- `gold_page`, `gold_section`
- `review_status`, `evidence_type`

지원 근거 형식은 `text`이며 `figure`, `table`, `formula`, `image`는 제외한다.

## 공개 및 저장 정책

공개 저장소에 포함할 수 있는 항목:

- 출처 URL과 논문 식별자
- 라이선스와 메타데이터
- 전처리, 검색, 평가, API 코드
- 재현 가능한 설정과 허용된 평가 산출물

공개 저장소에서 제외할 항목:

- 재배포 권한을 확인하지 않은 PDF
- QASPER 원본 전체 파일
- 사용자 업로드 파일
- API 키와 비밀 값
- 불필요한 원본 데이터

`.gitignore`는 다음 경로를 제외한다.

```gitignore
data/raw/qasper/
data/raw/papers/*.pdf
data/runtime/
```

논문별 최종 출처와 라이선스 판정은 `data/metadata/paper_manifest.csv`와 `data/metadata/license_review.md`를 기준으로 한다.
