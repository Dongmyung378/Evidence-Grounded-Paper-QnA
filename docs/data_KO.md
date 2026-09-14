# 데이터와 라이선스

[English](data.md) | [한국어](data_KO.md)

## 코퍼스

로컬 코퍼스에는 영어 논문 PDF 30편이 있습니다. 모두 텍스트 추출과 수집 검증을 통과했습니다. 공개 저장소에는 재현에 필요한 메타데이터와 처리 텍스트 산출물만 두고, 논문마다 재배포 조건이 다르므로 원본 PDF는 로컬에 유지합니다.

`data/metadata/paper_manifest.csv`에는 논문 제목, 원문 URL, 분야, 추출 상태, 페이지 수와 검토한 라이선스 상태를 기록합니다. 원문 URL은 출처 정보이며 PDF 재배포 권한을 자동으로 의미하지 않습니다.

## 추적성

수집 파이프라인은 다음 연결을 유지합니다.

```text
논문 PDF -> paper_id -> 페이지 -> page_id -> 절 -> chunk_id -> evidence_id
```

처리한 페이지는 `data/processed/pages.jsonl`, 검색 청크는 `data/processed/chunks.jsonl`, 파이프라인 수치는 `data/processed/pipeline_manifest.json`에 있습니다. API가 반환하는 근거에는 페이지, 절, 원문과 청크 식별자가 포함됩니다.

## 평가 데이터

`questions.jsonl`과 `gold_evidence.jsonl`에는 논문 001-010의 검증 검색 문항 40개가 있습니다. 답변 검토는 `answer_review_manifest.json`의 고정 20문항을 사용합니다. 별도의 비근거 질문으로 답변 거절 성능을 평가합니다. 주장 가능한 범위는 [평가 문서](evaluation_KO.md)에 정리했습니다.

## 라이선스 경계

- 원본 PDF는 Git에서 제외하며 Docker 이미지에도 복사하지 않습니다.
- 사용자 업로드는 로컬 실행 볼륨에만 남고 Git에서 제외합니다.
- 번역 구성요소 `facebook/nllb-200-distilled-600M`은 CC-BY-NC-4.0이므로 패키지 데모를 비상업 용도로 명시합니다.
- 원본 논문을 배포하기 전에는 해당 버전의 라이선스를 개별적으로 다시 확인해야 합니다.
