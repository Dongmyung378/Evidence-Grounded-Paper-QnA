# 별도 Linux 환경 재현

[English](README.md) | [한국어](README_KO.md)

논문 네 편에서 사전에 고른 언어 출력 12개의 답변, 원문 대응과 인용 출처가 Windows 결과와 정확히 일치했습니다. 논문 013의 두 부정 질문 대응 그룹도 포함했습니다. 전체 논문 30편에서 밀집 임베딩을 새로 생성했습니다. 모델 실행은 Python 3.12.15, Torch 2.5.1 CPU 및 Torch 스레드 두 개를 사용했으며, 네트워크를 차단한 Linux 컨테이너에서 읽기 전용 고정 모델 캐시를 활용했습니다. 물리 호스트는 Windows 실행과 같습니다.

[출력](linux-predictions.json), [비교 및 선정 범위](comparison.json)와 [측정 당시 실행기](measured_runner.py)에 표본을 보존합니다. 실행기는 `/repo`에 복원된 측정 당시 구현, 오프라인 모델과 쓰기 가능한 `/repo/outputs`를 사용합니다. 실행 소스의 해시는 최종 측정 당시 파일 80개와 모두 일치합니다. 이후 외부 복원 도구와 산출물 검증기를 수정했지만 추론 소스는 변경하지 않았습니다.

에이전트가 수행한 환경 재현입니다. 독립적인 사람의 사실 검토, 전체 출력 360개의 모델 재현이나 지연시간 벤치마크는 아닙니다. 시간과 재정렬 점수의 부동소수점 차이는 정확한 출력 비교에서 제외했습니다. 공개 평가 분할 전체를 다시 실행하려면 저장소 최상위에서 `scripts/replay_corpus_benchmark.py --variant final --split holdout --offline --output outputs/final-replay.json`을 사용합니다.
