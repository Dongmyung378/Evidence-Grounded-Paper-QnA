# 문서 안내

최상단 `README.md`와 `README_KO.md`는 프로젝트 소개와 실행 방법을 제공한다. 상세 문서는 구현 일지 대신 기능, 검증, 한계를 기준으로 구성한다.

## 프로젝트 문서

- [프로젝트 범위](project/scope.md)
- [기능 및 비기능 요구사항](project/requirements.md)
- [데이터 출처와 취급 정책](project/data_sources.md)

## API 문서

- [PDF 업로드와 분석 API](api/upload_analysis.md)
- [질문, 논문 결과, 상태 API](api/question_answer.md)
- [전체 API 통합 게이트](api/integration.md)

## UI 문서

- [Streamlit interface in English](ui/interface.md)
- [Streamlit 사용자 화면](ui/interface_KO.md)

## 실행 및 재현 문서

- [Docker Compose local run](deployment/docker.md)
- [Docker Compose 로컬 실행](deployment/docker_KO.md)

## 품질 및 실행 검토

- [답변 품질 및 개선 검토](reviews/improvements.md)
- [로컬 실행 성능 개선](reviews/runtime_performance_KO.md)
- [Local runtime performance](reviews/runtime_performance.md)

평가 수치와 실행 증거는 `data/evaluation/`에 있으며, 각 파일의 역할은 [평가 데이터 안내](../data/evaluation/README.md)에 정리되어 있다.
