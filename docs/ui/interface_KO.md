# Streamlit 사용자 화면

[English](interface.md) | [한국어](interface_KO.md)

## 사용자 흐름

`ui/app.py`의 Streamlit 화면은 `ui/api_client.py`를 통해 FastAPI와 통신한다. 파싱, 검색, 재정렬, 답변 생성, 인용 검증은 백엔드에서 처리한다.

화면에서 다음 기능을 제공한다.

- 영어와 한국어 화면 전환
- 최대 20 MiB의 텍스트 기반 PDF 한 편 업로드
- 분석 대기, 진행, 완료, 실패 상태 표시
- 전체 페이지, 텍스트 페이지, 청크, 경고 수 표시
- 추출한 초록과 식별한 절 제목 표시
- 분석 완료 후에만 활성화되는 질문 입력
- 답변, 거절 사유, 단계별 실행 시간 표시
- 답변 옆 원문 근거 패널 표시
- 한영 업로드 검증 메시지와 파싱 실패 복구 상태

## 근거 표시

인용한 근거는 각각 테두리가 있는 카드로 표시한다. 카드에는 인용 순번, 페이지, 절, 논문 원문, 청크 ID가 포함된다. 답변 패널에는 인용한 근거 수를 표시한다. 백엔드가 근거 부족으로 판단하면 근거 카드를 만들지 않고 거절 사유만 보여준다.

이 구성은 답변을 논문 원문까지 추적할 수 있게 하면서 백엔드의 검증된 근거 계약을 그대로 사용한다. 현재 로컬 생성 모델의 답변 품질이 운영 수준이라는 의미는 아니다.

## 오류 처리와 복구

화면과 FastAPI는 동일한 20 MiB 제한을 적용한다. 파일 크기, 확장자, 콘텐츠 형식, PDF 헤더는 업로드 전에 확인하고, PDF 전체의 가독성과 텍스트 추출 가능 여부는 백엔드에서 검사한다. 큰 파일, 잘못되거나 암호화된 PDF, 텍스트 없음, 대기열 혼잡, 시간 초과, 파싱 실패, 서비스 중단에 대한 한국어와 영어 메시지를 제공한다. 알 수 없는 백엔드 오류는 내부 내용을 노출하지 않고 안전한 일반 메시지로 바꾼다.

파싱에 실패하면 진행 표시를 제거하고 분석 상태를 오류로 종료하며 질문 입력은 잠긴 상태로 유지한다. 파일 선택과 업로드 버튼은 계속 사용할 수 있으므로 화면을 새로고침하지 않고 다른 PDF로 재시도할 수 있다.

## 로컬 실행

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

두 번째 터미널에서 Streamlit을 실행한다.

```bash
python -m streamlit run ui/app.py
```

브라우저에서 `http://127.0.0.1:8501`을 연다. FastAPI 주소가 다르면 화면 왼쪽 설정이나 `PAPER_QNA_API_URL` 환경변수에서 변경할 수 있다.

## 검증

브라우저 게이트는 실제 로컬 FastAPI 서버, Streamlit 서버, 헤드리스 Chromium 브라우저를 사용한다. 첫 번째 게이트는 PDF 업로드, 분석, 개요, 질문 준비 상태를 검증한다. 두 번째 게이트는 답변 가능한 질문을 보내고 답변과 원문 근거가 나란히 표시되는지, 각 근거에 페이지, 절, 원문, 청크 ID가 있는지 확인한다.

| 검증 항목 | 결과 |
|---|---:|
| 회귀 및 UI 테스트 | 45개 통과 |
| 브라우저 PDF 업로드와 분석 | 통과 |
| 브라우저 질문 전송 | 통과 |
| 답변과 근거 나란히 표시 | 통과 |
| 표시한 근거 카드 | 4개 |
| 페이지, 절, 원문, 청크 ID | 통과 |
| 큰 파일과 잘못된 PDF의 API 전 차단 | 통과 |
| 파싱 실패 메시지와 진행 상태 종료 | 통과 |
| 파싱 실패 후 정상 파일 업로드 | 통과 |
| 시드 | 378 |
| 고정 평가 입력 변경 | 없음 |
| 임시 런타임 잔존 | 없음 |

```bash
python -B scripts/validate_day36.py
python -B scripts/validate_evidence_ui.py
python -B scripts/validate_error_recovery_ui.py
```

기계 판독 결과는 `data/evaluation/day36_ui_results.json`, `data/evaluation/evidence_ui_results.json`, `data/evaluation/error_recovery_ui_results.json`에 저장한다. 브라우저 평가기를 다시 실행하려면 Python 의존성 외에 Node.js, Playwright, 로컬 Chromium 브라우저가 필요하다.
