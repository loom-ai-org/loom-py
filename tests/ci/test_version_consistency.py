"""VERSION is the version, and this is what makes that true rather than aspirational.

The four published packages pin each other by EXACT version, in a circle (packaging/README.md), so
one release number is written in eleven places in two spellings across five files. `VERSION` plus
`packaging/version.py --set` is how it gets written; this test is why nothing else can. It is the
half that catches a hand edit -- a bumped `pyproject.toml` with the README's ARMv6 wheel URL left
behind publishes a 404, and only for the users with the smallest boards.

It runs everywhere `tests/ci` runs: ci.yml on every push, and INSIDE every wheel cibuildwheel builds
(`test-command = "... pytest {project}/tests/ci -q"`), which is the last moment before an upload.

Deliberately imports no `loom`: this is a property of the source tree, and it should still fail on a
checkout where the extension was never built.
"""

import importlib.util
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


def _tool():
    """packaging/version.py, loaded by path -- `packaging/` is release tooling, not an importable package."""
    path = REPO / "packaging" / "version.py"
    assert path.is_file(), f"{path} is missing; it is the only thing that should write a version"
    spec = importlib.util.spec_from_file_location("loom_packaging_version", path)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE it is executed, and not as a tidiness: `Site` is a dataclass in a module
    # with `from __future__ import annotations`, so `@dataclass` resolves its field types by looking
    # its own module up in `sys.modules` -- which for a module loaded by path is not there yet, and
    # the failure is an AttributeError inside dataclasses at COLLECTION time.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


def test_version_file_holds_a_version_the_tool_accepts():
    version = (REPO / "VERSION").read_text().strip()
    assert TOOL.READABLE.match(version), (
        f"VERSION holds {version!r}; packaging/version.py cannot derive a PEP 440 pin from it"
    )


@pytest.mark.parametrize("site", TOOL.SITES, ids=lambda s: f"{s.path}:{s.what}")
def test_every_derived_copy_agrees_with_the_version_file(site):
    """One assertion per place the version is written, so a failure names the file and the reason."""
    version = (REPO / "VERSION").read_text().strip()
    text = (REPO / site.path).read_text()
    found = list(re.finditer(site.pattern, text))

    # ZERO MATCHES IS THE INTERESTING FAILURE, not a nuisance one: it means the file changed shape
    # and the SITES table did not, and a table that matches nothing would let `--check` pass while
    # checking nothing.
    assert len(found) == 1, (
        f"{site.path}: {len(found)} matches for {site.what}, expected exactly 1 -- the file moved "
        "and the SITES table in packaging/version.py did not follow it"
    )
    assert found[0].group("v") == site.wanted(version), (
        f"{site.path} says {found[0].group('v')} for {site.what}, VERSION says {version} -- run "
        "`python packaging/version.py --set <version>` rather than editing it"
    )


def test_the_two_spellings_are_derived_the_way_pep_440_normalises():
    """`version = "1.0.0-rc9"` and `== 1.0.0rc9` are the same version; a pin in the first spelling
    resolves against nothing. The tool derives the second by dropping one hyphen, and that is only
    safe because `packaging` agrees -- so check it against `packaging` where it is installed (it is a
    pytest dependency, so that is here) rather than trusting the regex."""
    packaging_version = pytest.importorskip("packaging.version")
    for readable in ("1.0.0", "1.0.0-rc9", "1.0.0-rc10", "2.3.4-a1", "2.3.4-b2"):
        assert TOOL.READABLE.match(readable), readable
        assert TOOL.pep440(readable) == str(packaging_version.Version(readable))
