"""Where the wheel finds its ggml backends, and what happens when it does not.

This is not incidental plumbing -- with `GGML_BACKEND_DL` it is the difference between a working
package and one that cannot do anything at all. Every backend is a shared library loaded at run time,
INCLUDING the CPU, so a wheel that ships no .so, or that ships them where nothing looks, has an empty
device registry: `device="cpu"` fails, `device="auto"` fails, and the error is not "no GPU" but "no
devices". ggml's own default search (the executable's directory, then the current directory) never
finds them, because inside an interpreter the executable is `python`.

So the first test here is the wheel's smoke test, and the rest pin the discovery rule that makes an
accelerator installable as a separate package.
"""
import os
import pathlib
import sys

import pytest

import loom


def test_a_cpu_device_is_available():
    """The one that fails loudly if the packaging broke.

    Any of: the .so files were not installed beside the package, `$ORIGIN` did not make it into the
    RPATH so libloom_engine.so could not find libggml-base.so, or the search path was never
    registered. All of them land here as an empty or CPU-less list.
    """
    devices = loom.devices()
    assert devices, (
        "no ggml devices at all -- with GGML_BACKEND_DL that means no backend .so was found, not "
        "that the machine lacks an accelerator"
    )
    assert any(d["is_cpu"] for d in devices), [d["name"] for d in devices]


def test_devices_report_names_and_descriptions():
    for device in loom.devices():
        assert device["name"]
        assert isinstance(device["is_cpu"], bool)
        # Reported only by devices that track it; 0 is a legitimate answer, not a missing key.
        assert device["memory_total"] >= 0


def _recorded_paths(monkeypatch, extra_sys_path=(), env=None):
    """Re-run discovery with the engine call captured instead of performed.

    Registration is idempotent on the engine side (ggml dedupes on the registration pointer), so
    calling `_register_backend_paths` again in-process would be harmless -- but capturing it is what
    lets these tests assert the RULE rather than its effect, which is the part that has to keep
    working for a backend package that is not installed here.
    """
    seen = []
    monkeypatch.setattr(loom._loom, "add_backend_search_path", seen.append)
    monkeypatch.setenv("LOOM_BACKEND_DIR", env or "")
    for entry in extra_sys_path:
        monkeypatch.syspath_prepend(entry)
    loom._register_backend_paths()
    return seen


def test_the_package_directory_is_always_searched(monkeypatch):
    """Where the base wheel's own libggml-base.so and libggml-cpu-*.so live.

    `resolve()` rather than `abspath()`, and the difference is not pedantry -- it is the difference
    between this test passing and failing on macOS. `_register_backend_paths` resolves, because ggml
    is handed a real path rather than one it has to interpret; `abspath` does not follow symlinks.
    On Linux the two agree and the distinction never showed. On macOS BOTH standard temporary roots
    are symlinks -- `/tmp` -> `/private/tmp` and `/var` -> `/private/var` -- so a package installed
    into a venv under either (which is every wheel test step, cibuildwheel's included) compares
    `/tmp/.../loom` against the recorded `/private/tmp/.../loom` and fails on a path that is the
    same directory.
    """
    package_dir = str(pathlib.Path(loom.__file__).resolve().parent)
    assert package_dir in _recorded_paths(monkeypatch)


def test_an_installed_backend_package_is_found(tmp_path, monkeypatch):
    """The `loom-py-rt-vulkan` shape: a `loom_rt_*` directory anywhere on sys.path.

    Discovery is a directory scan and not an import, deliberately -- a backend package that fails to
    import for an unrelated reason must not be able to take the base package down with it -- so this
    fixture is a bare directory with no `__init__.py` and is still expected to be found.
    """
    (tmp_path / "loom_rt_madeup").mkdir()
    recorded = _recorded_paths(monkeypatch, extra_sys_path=[str(tmp_path)])
    assert str((tmp_path / "loom_rt_madeup").resolve()) in recorded


def test_a_file_named_like_a_backend_package_is_not_a_directory(tmp_path, monkeypatch):
    (tmp_path / "loom_rt_notadir").write_text("")
    recorded = _recorded_paths(monkeypatch, extra_sys_path=[str(tmp_path)])
    assert not any("loom_rt_notadir" in path for path in recorded)


def test_the_environment_override_comes_first(tmp_path, monkeypatch):
    """`$LOOM_BACKEND_DIR` is for a backend no package installed -- a build tree, a hand-built .so.

    First in the list, matching the resolution order this project uses everywhere: the explicit
    override, then what the installation can work out for itself.
    """
    first, second = tmp_path / "one", tmp_path / "two"
    recorded = _recorded_paths(
        monkeypatch, env=os.pathsep.join([str(first), str(second)])
    )
    assert recorded[:2] == [str(first), str(second)]


def test_an_empty_environment_override_is_not_a_path(monkeypatch):
    """Exporting it blank must not register the current directory."""
    assert "" not in _recorded_paths(monkeypatch, env=os.pathsep)
