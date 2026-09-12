"""Validate the repository's local-only portfolio delivery boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def main() -> None:
    scope = read("docs/project/scope.md")
    requirements = read("docs/project/requirements.md")
    readme = read("README.md")
    readme_ko = read("README_KO.md")
    docs_index = read("docs/README.md")
    docker_en = read("docs/deployment/docker.md")
    docker_ko = read("docs/deployment/docker_KO.md")
    gitignore = read(".gitignore")
    compose = read("compose.yaml")

    assert "로컬 포트폴리오 데모" in scope
    assert "Docker Compose가 최종 제공 환경" in scope
    assert "공개 서버 운영과 공개 URL은 완료 조건에 포함하지 않는다" in requirements
    assert "local portfolio demo" in readme
    assert "public server hosting and a public URL are intentionally outside" in readme
    assert "로컬 포트폴리오 데모" in readme_ko
    assert "공개 서버 운영과 공개 URL은 프로젝트 범위에서 의도적으로 제외" in readme_ko
    assert "## 실행 및 재현 문서" in docs_index
    assert "public server hosting is intentionally outside" in docker_en
    assert "공개 서버 운영은 프로젝트 범위에서 의도적으로 제외" in docker_ko
    assert "local_notes/" in gitignore
    assert "  backend:" in compose and "  ui:" in compose

    forbidden_claims = {
        "docs/project/requirements.md": "공개 URL 배포가 가능해야 한다",
        "README.md": "deployment to an internet-facing server has not",
        "README_KO.md": "인터넷에 공개된 서버 배포는 아직 수행하지 않았습니다",
    }
    for path, phrase in forbidden_claims.items():
        assert phrase not in read(path), f"Obsolete deployment requirement remains: {path}"

    print("Local portfolio delivery scope validation passed")
    print("delivery=Docker Compose local demo public_server=false public_url=false")
    print("portfolio_evidence=screenshots video_optional seed=378")


if __name__ == "__main__":
    main()
