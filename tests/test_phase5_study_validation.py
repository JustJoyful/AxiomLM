import pytest

from scripts import validate_phase5_study as v


@pytest.mark.asyncio
async def test_validator_refuses_existing_output(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        await v.validate_study([], output=path)
    assert path.read_text() == "original"


@pytest.mark.asyncio
async def test_unavailable_catalog_stays_a_failed_case(tmp_path, monkeypatch):
    monkeypatch.setattr(v.config, "VECTOR_STORE", tmp_path / "empty")
    report = await v.validate_study(["unknown"], output=tmp_path / "report.json")
    assert report["accepted"] is False
    assert report["books"][0]["status"] == "error"
    assert report["human_review"] is None


@pytest.mark.asyncio
async def test_validator_uses_temporary_state_indexes_and_blank_verdicts(
    tmp_path, monkeypatch
):
    from src import study
    from src.study_state import StudyStore
    from tests.test_study_state import catalog_root

    catalog_root(tmp_path)
    originals = {str(p): p.read_bytes() for p in tmp_path.rglob("*.sqlite3")}
    monkeypatch.setattr(v.config, "VECTOR_STORE", tmp_path)
    monkeypatch.setattr(v, "identities", lambda paths: {"runtime": "unit fixture"})
    monkeypatch.setattr(v, "load_chat_counter", lambda model: object())

    class Client:
        async def achat(self, *a, **k):
            raise AssertionError("unit validator must not call Ollama")

        def unload_and_wait(self):
            pass

    monkeypatch.setattr(v, "OllamaClient", Client)
    state_paths = []

    class Store(StudyStore):
        def __init__(self, path):
            state_paths.append(path)
            super().__init__(path)

    monkeypatch.setattr(v, "StudyStore", Store)

    async def generate(section, count, **kwargs):
        refs = study.section_evidence(section)
        return study.PracticeSet(
            section,
            tuple(
                study.PracticeQuestion(f"Question {i}?", (refs[0].source_id,))
                for i in range(count)
            ),
            refs,
        )

    monkeypatch.setattr(study, "generate_practice", generate)
    report = await v.validate_study(["sicp"], output=tmp_path / "report.json")
    row = report["books"][0]
    assert row["status"] == "measured" and row["originals_unchanged"]
    assert all(
        c["human_checked"] is None and c["human_supported"] is None
        for c in row["cases"]
    )
    assert all(not path.exists() for path in state_paths)
    assert all(
        __import__("pathlib").Path(p).read_bytes() == text
        for p, text in originals.items()
    )
