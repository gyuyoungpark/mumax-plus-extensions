"""Source coverage must follow modular engine layouts, not kernel names."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "runs"))
import _provenance as P


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "REPO", str(tmp_path))
    monkeypatch.setattr(P, "PHYSICS_DIR", str(tmp_path / "src" / "physics"))
    (tmp_path / "src" / "physics").mkdir(parents=True)
    for name in ("CMakeLists.txt", "setup.py", "pyproject.toml"):
        (tmp_path / name).touch()
    return tmp_path


def make_file(repo, relative):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("// fixture\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("layout", ["extension", "extensions"])
def test_modular_extension_sources_and_build_rules_are_hashed(repo, layout):
    module = f"{layout}/new-physics"
    files = [make_file(repo, f"{module}/{name}") for name in
             ("CMakeLists.txt", "src/new.cu", "include/new.hpp", "cmake/flags.cmake")]
    discovered = {Path(p) for p in P.compiled_source_files()}
    assert set(files) <= discovered
    before = P._digest({str(p): P.sha256_file(p) for p in discovered})
    files[2].write_text("// changed header\n", encoding="utf-8")
    after = P._digest({str(p): P.sha256_file(p) for p in discovered})
    assert before != after


def test_new_core_subdirectory_is_not_silently_omitted(repo):
    source = make_file(repo, "src/new-solver/nested/solver.cpp")
    rule = make_file(repo, "src/new-solver/settings.cmake")
    assert {source, rule} <= {Path(p) for p in P.compiled_source_files()}


def test_legacy_non_build_extension_is_not_claimed_as_compiled(repo):
    archive = make_file(repo, "extensions/legacy/src/obsolete.cu")
    assert archive not in {Path(p) for p in P.compiled_source_files()}
