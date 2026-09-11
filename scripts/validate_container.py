"""Validate the saved Docker Compose acceptance record and packaging files."""

import json

from evaluate_container import OUTPUT, ROOT, implementation_hashes


def main() -> None:
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["seed"] == 378
    assert result["requirement"] == (
        "docker compose up runs the backend and UI local demo"
    )
    assert result["implementation_sha256"] == implementation_hashes(), (
        "Rerun evaluate_container.py after changing container implementation"
    )
    assert result["temporary_stack_removed"] is True
    assert result["quality_claim"] == (
        "local Docker packaging only; public deployment is not verified"
    )

    services = result["services"]
    assert set(services) == {"backend", "ui"}
    for service in services.values():
        assert service["state"] == "running"
        assert service["health"] == "healthy"

    http = result["http"]
    assert http["backend_health_status"] == 200
    assert http["backend_service"] == "evidence-grounded-paper-qna"
    assert http["ui_health_status"] == 200
    assert http["ui_page_status"] == 200

    configuration = result["configuration"]
    assert configuration["ui_backend_url"] == "http://backend:8000"
    assert configuration["runtime_volume"] == "/var/lib/paper-qna"
    assert configuration["model_cache_volume"] == "/var/cache/huggingface"
    assert configuration["secrets_recorded"] is False
    assert all(result["security"].values())

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert ".env\n" in gitignore
    for private_path in (".env", "local_notes/", "data/"):
        assert private_path in dockerignore

    print("Docker Compose local demo gate passed")
    print("services=backend:healthy ui:healthy HTTP=200+200 seed=378")
    print("cleanup=containers+network+test-volumes removed")


if __name__ == "__main__":
    main()
