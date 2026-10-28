"""Working rule 1: truth never leaks into perception or control."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "meatpick"
FORBIDDEN = "meatpick.sim"


def _imports(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            found.append(mod)
            if mod == "meatpick":  # `from meatpick import sim`
                found += [f"meatpick.{a.name}" for a in node.names]
    return found


@pytest.mark.parametrize("package", ["perception", "control"])
def test_package_never_imports_the_simulator(package):
    files = list((ROOT / package).rglob("*.py"))
    assert files
    for f in files:
        bad = [m for m in _imports(f) if m == FORBIDDEN or m.startswith(FORBIDDEN + ".")]
        assert not bad, f"{f.relative_to(ROOT)} imports {bad}"


def test_the_checker_itself_catches_a_violation(tmp_path):
    bad = tmp_path / "x.py"
    bad.write_text("from meatpick.sim.world import World\nfrom meatpick import sim\n")
    assert _imports(bad) == ["meatpick.sim.world", "meatpick", "meatpick.sim"]
