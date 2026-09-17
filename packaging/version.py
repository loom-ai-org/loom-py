#!/usr/bin/env python3
"""The version of every package in this repo, in one file, plus the tool that propagates it.

WHY THIS EXISTS. The four published packages -- `loom-py-rt` and the `-cuda`/`-vulkan`/`-metal`
backends -- pin each other by EXACT version, in a circle, because a backend `.so` links
`libggml-base.so` and ggml makes no ABI promise across revisions (packaging/README.md, "Two things
that are not negotiable"). A circular exact-pin set means a release is not four independent numbers
but one number written in eleven places, in TWO spellings, across FIVE files -- and bumping any subset
publishes a package that resolves against a version nobody released. That arithmetic grew from seven
strings to ten to eleven as packages were added (rt-metal, then the ARMv6 wheel URL in README.md),
which is the shape of a thing that should not be counted by hand at all.

So `VERSION` at the repo root is the version, and everything else is derived:

    python packaging/version.py              # what is the version, and is every copy in step?
    python packaging/version.py --set 1.0.0-rc11
    python packaging/version.py --print          # 1.0.0-rc11   (readable, for `version =`)
    python packaging/version.py --print-pep440   # 1.0.0rc11    (normalised, for a `==` pin)

`--set` is the only thing that should ever write a version into a pyproject, and
`tests/ci/test_version_consistency.py` runs the check on every push, in CI and inside every wheel
cibuildwheel builds -- so a hand edit that misses a file fails the suite instead of shipping.

WHY THE DERIVED COPIES ARE STILL LITERAL TEXT rather than dynamic metadata read out of this file at
build time: scikit-build-core can do that for `version`, but the pins live in `dependencies` and
`optional-dependencies`, and the three backend wheels are built by cibuildwheel from a STAGED tree
(packaging/stage.py) that contains only the package directory -- a build-time read of a repo-root
file is not available to the build that most needs it. Static text that a test proves correct is
worth more here than dynamic text that is correct only where it happens to resolve.

THE TWO SPELLINGS ARE NOT INTERCHANGEABLE. `version = "1.0.0-rc11"` is the readable form PEP 621
accepts and normalises; a dependency specifier must carry PEP 440's normal form, `== 1.0.0rc11`, or
it resolves against nothing. This tool derives the second from the first, and the test cross-checks
the derivation against `packaging.version.Version` wherever that library is importable.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO / "VERSION"

# The readable spelling, deliberately narrow: three numeric components and an optional
# `-a`/`-b`/`-rc` prerelease. Every version this project has published fits it, and anything that
# does not (a `.post`, a local segment, an epoch) needs `pep440()` below looked at rather than the
# pattern widened, since the normalisation rules differ per segment.
READABLE = re.compile(r"^\d+\.\d+\.\d+(?:-(?:a|b|rc)\d+)?$")


def pep440(readable: str) -> str:
    """`1.0.0-rc9` -> `1.0.0rc9`. The hyphen is a separator PEP 440 drops; nothing else changes."""
    return re.sub(r"-(a|b|rc)(\d+)$", r"\1\2", readable)


@dataclass(frozen=True)
class Site:
    """One place the version is written, and which spelling belongs there.

    `pattern` must match EXACTLY ONCE in `path`. That is the load-bearing half: a pattern that
    silently stops matching -- because a file was reformatted or a package renamed -- would make
    `--check` pass by checking nothing, which is the failure mode this whole file exists to prevent.
    """

    path: str
    pattern: str
    spelling: str  # "readable" or "pep440"
    what: str

    def wanted(self, readable: str) -> str:
        return readable if self.spelling == "readable" else pep440(readable)


SITES = (
    Site("pyproject.toml", r'(?m)^version = "(?P<v>[^"]+)"$', "readable", "loom-py-rt's own version"),
    Site("pyproject.toml", r'"loom-py-rt-vulkan == (?P<v>[^"]+)"', "pep440", "the [vulkan] extra's pin"),
    Site("pyproject.toml", r'"loom-py-rt-cuda == (?P<v>[^"]+)"', "pep440", "the [cuda] extra's pin"),
    Site("pyproject.toml", r'"loom-py-rt-metal == (?P<v>[^"]+)"', "pep440", "the [metal] extra's pin"),
    Site("packaging/rt-cuda/pyproject.toml", r'(?m)^version = "(?P<v>[^"]+)"$', "readable", "loom-py-rt-cuda's own version"),
    Site("packaging/rt-cuda/pyproject.toml", r'"loom-py-rt == (?P<v>[^"]+)"', "pep440", "its pin back on the base wheel"),
    Site("packaging/rt-vulkan/pyproject.toml", r'(?m)^version = "(?P<v>[^"]+)"$', "readable", "loom-py-rt-vulkan's own version"),
    Site("packaging/rt-vulkan/pyproject.toml", r'"loom-py-rt == (?P<v>[^"]+)"', "pep440", "its pin back on the base wheel"),
    Site("packaging/rt-metal/pyproject.toml", r'(?m)^version = "(?P<v>[^"]+)"$', "readable", "loom-py-rt-metal's own version"),
    Site("packaging/rt-metal/pyproject.toml", r'"loom-py-rt == (?P<v>[^"]+)"', "pep440", "its pin back on the base wheel"),
    # THE ONE THAT IS NOT A PACKAGE AND IS THE EASIEST TO MISS. PyPI refuses a `linux_armv6l` tag, so
    # the Pi Zero wheel is a GitHub release asset installed by URL -- `releases/latest/download/` is
    # the stable half of that URL and the VERSION IS IN THE FILENAME, so a missed bump here is a 404
    # the moment the release ships, and only for the users with the smallest boards.
    Site("README.md", r"loom_py_rt-(?P<v>[^-]+)-cp311-cp311-linux_armv6l\.whl", "pep440",
         "the ARMv6 wheel's install URL"),
)


def read_version() -> str:
    if not VERSION_FILE.is_file():
        sys.exit(f"{VERSION_FILE} is missing -- it is the version, so there is nothing to derive from")
    readable = VERSION_FILE.read_text().strip()
    if not READABLE.match(readable):
        sys.exit(
            f"{VERSION_FILE} holds {readable!r}, which is not `N.N.N` or `N.N.N-rcN` -- see the "
            "READABLE pattern in packaging/version.py before widening it"
        )
    return readable


def _matches(site: Site) -> list[re.Match[str]]:
    path = REPO / site.path
    if not path.is_file():
        sys.exit(f"{site.path} does not exist, but packaging/version.py still expects to write {site.what} into it")
    found = list(re.finditer(site.pattern, path.read_text()))
    if len(found) != 1:
        sys.exit(
            f"{site.path}: expected exactly one match for {site.what} ({len(found)} found).\n"
            "The file changed shape and the SITES table in packaging/version.py did not. Fix the "
            "table -- a pattern that matches nothing makes --check pass without checking anything."
        )
    return found


def check(readable: str, expect_tag: str | None = None) -> int:
    drift = []
    for site in SITES:
        found = _matches(site)[0].group("v")
        wanted = site.wanted(readable)
        mark = " " if found == wanted else "!"
        if found != wanted:
            drift.append((site, found, wanted))
        print(f" {mark} {site.path:<36} {found:<12} {site.what}")

    if expect_tag:
        tag = expect_tag[1:] if expect_tag.startswith("v") else expect_tag
        if tag != readable:
            drift.append(("the git tag", tag, readable))
            print(f" ! git tag{'':<27} {tag:<12} the release this is being published as")

    if drift:
        print(
            f"\n{len(drift)} place(s) disagree with VERSION ({readable}).\n"
            "Run `python packaging/version.py --set <version>`; do not edit them by hand.",
            file=sys.stderr,
        )
        return 1
    print(f"\nall {len(SITES)} derived copies agree with VERSION ({readable} / {pep440(readable)})")
    return 0


def set_version(readable: str) -> int:
    if not READABLE.match(readable):
        sys.exit(f"{readable!r} is not `N.N.N` or `N.N.N-rcN`")

    VERSION_FILE.write_text(readable + "\n")
    for site in SITES:
        path = REPO / site.path
        match = _matches(site)[0]
        wanted = site.wanted(readable)
        if match.group("v") == wanted:
            continue
        text = path.read_text()
        start, end = match.span("v")
        path.write_text(text[:start] + wanted + text[end:])
        print(f"   {site.path:<36} {match.group('v'):>12} -> {wanted:<12} {site.what}")

    print(f"\nVERSION is {readable}; every derived copy rewritten. Commit them TOGETHER -- a tree "
          "where they disagree is one whose packages resolve against a release nobody cut.")
    return check(readable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--set", metavar="VERSION", help="write this version into VERSION and every derived copy")
    group.add_argument("--check", action="store_true", help="verify every derived copy (the default)")
    group.add_argument("--print", dest="show", action="store_true", help="print the readable version")
    group.add_argument("--print-pep440", dest="show_pep440", action="store_true",
                       help="print the PEP 440 normal form, which is what a `==` pin needs")
    parser.add_argument("--expect-tag", metavar="TAG",
                        help="also require this git tag to name the same version (the release workflow passes it)")
    args = parser.parse_args()

    if args.set:
        raise SystemExit(set_version(args.set))
    if args.show:
        print(read_version())
        return
    if args.show_pep440:
        print(pep440(read_version()))
        return
    raise SystemExit(check(read_version(), args.expect_tag))


if __name__ == "__main__":
    main()
