"""Validate the public portfolio structure and local-only boundaries."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAY_PATTERN = re.compile(r"(?i)(?:^|[^a-z])day[_ -]?\d+")
SCAN_SUFFIXES = {".md", ".py", ".json", ".yaml", ".yml"}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def main():
    required_pairs = [
        ("README.md", "README_KO.md"),
        ("docs/README.md", "docs/README_KO.md"),
        ("docs/architecture.md", "docs/architecture_KO.md"),
        ("docs/evaluation.md", "docs/evaluation_KO.md"),
        ("docs/failure_cases.md", "docs/failure_cases_KO.md"),
        ("docs/data.md", "docs/data_KO.md"),
        ("docs/deployment/docker.md", "docs/deployment/docker_KO.md"),
        ("data/README.md", "data/README_KO.md"),
        ("data/evaluation/README.md", "data/evaluation/README_KO.md"),
        ("scripts/README.md", "scripts/README_KO.md"),
    ]
    for english, korean in required_pairs:
        assert (ROOT / english).is_file() and (ROOT / korean).is_file()
        assert "[English]" in read(english) and "[한국어]" in read(english)
        assert "[English]" in read(korean) and "[한국어]" in read(korean)

    screenshots = ROOT / "docs" / "assets" / "screenshots"
    expected_screenshots = {
        "01-upload.png",
        "02-analysis-complete.png",
        "03-english-answer-evidence.png",
        "04-korean-answer-evidence.png",
        "docker.png",
    }
    assert {path.name for path in screenshots.glob("*.png")} == expected_screenshots
    assert all((screenshots / name).stat().st_size > 0 for name in expected_screenshots)

    failure_en = read("docs/failure_cases.md")
    failure_ko = read("docs/failure_cases_KO.md")
    failure_ids = {
        "q-001-en",
        "q-001-ko",
        "abs-test-006-ko",
        "q-017-ko",
        "q-015-en",
        "q-042-en",
        "q-042-ko",
        "q-043-ko",
        "q-041-ko",
        "q-034-en",
    }
    assert all(case_id in failure_en and case_id in failure_ko for case_id in failure_ids)
    assert "Citation support failures were 0" in failure_en
    assert "인용 지지 실패는 0건" in failure_ko

    excluded = {".git", "local_notes", "data/raw", "data/runtime"}
    violations = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if any(relative == item or relative.startswith(item + "/") for item in excluded):
            continue
        if DAY_PATTERN.search(relative):
            violations.append(relative)
        if path.suffix.lower() in SCAN_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            milestone_key = "roadmap" + "_day"
            if milestone_key in text:
                violations.append(relative)
    assert not violations, f"Milestone-only references remain: {sorted(set(violations))}"

    broken_links = []
    for document in ROOT.rglob("*.md"):
        relative = document.relative_to(ROOT).as_posix()
        if relative.startswith("local_notes/"):
            continue
        for target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            target = target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (document.parent / target).resolve().exists():
                broken_links.append(f"{relative} -> {target}")
    assert not broken_links, f"Broken local documentation links: {broken_links}"

    gitignore = read(".gitignore")
    compose = read("compose.yaml")
    assert "local_notes/" in gitignore
    assert "data/raw/papers/*.pdf" in gitignore
    assert "data/runtime/" in gitignore
    assert "  backend:" in compose and "  ui:" in compose
    assert "600 seconds" in read("README.md")
    assert "600초" in read("README_KO.md")
    print("Portfolio structure validation passed")
    print("docs=english-first+bilingual screenshots=5 milestone_files=0 broken_links=0")
    print("delivery=local Docker Compose public_server=false seed=378")


if __name__ == "__main__":
    main()
