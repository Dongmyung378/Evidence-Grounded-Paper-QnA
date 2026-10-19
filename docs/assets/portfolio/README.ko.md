# 포트폴리오 시각 자료

[English](README.md) | [한국어](README.ko.md)

보존한 한국어 SVG 그래프와 썸네일입니다. 수치는 `visual-data.json`에 기록한 기존 실험 자료와 체크섬을 근거로 합니다. 영문 문서는 같은 구조와 팔레트 및 수치를 유지한 영어 파생본을 연결합니다. 학습 모델과 데이터 분할, 선택 판단 및 원래 실험 보고서는 변경하지 않습니다.

생성기는 Python 기본 라이브러리와 포함된 표시용 데이터만 사용합니다. 그래프를 다시 생성하는 것은 모델 실행이나 과학적 평가 검증이 아닙니다.

```bash
python docs/assets/portfolio/render_visuals.py --input docs/assets/portfolio/visual-data.json --output docs/assets/portfolio
```

`visual-receipt.json`은 보존한 한국어 SVG를 표시용 데이터에 연결합니다. 영어 파생본과 [한국어 처리도](../paper-evidence.ko.svg)는 아래 명령으로 생성하고 대조합니다. `localization.json`에 언어별 문구를 기록하고 `localization-receipt.json`에 원본 및 파생본의 SHA-256을 연결합니다. 이 작업은 기존 사이트와 ZIP을 변경하거나 성능 측정을 다시 수행하지 않습니다.

```bash
python docs/assets/portfolio/render_localized_visuals.py
python docs/assets/portfolio/render_localized_visuals.py --check
```

## 시각 자료

- [paper-qna-thumbnail.svg](paper-qna-thumbnail.svg)
- [paper-qna-results.svg](paper-qna-results.svg)
