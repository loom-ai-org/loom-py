"""Getting a loom GGUF out of a HuggingFace repo.

Separate from `__init__.py` because it shares nothing with the engine: it is `huggingface_hub` and a
rule for picking a file, and keeping it here means `import loom` does not need the hub installed to
load a model from disk.
"""
from __future__ import annotations

import os
from pathlib import Path

GGUF_SUFFIX = ".gguf"


def _require_hub():
    try:
        import huggingface_hub  # noqa: PLC0415 -- optional, and only for this path
    except ImportError:
        raise ImportError(
            "loading from a HuggingFace repo needs `huggingface_hub`:\n"
            "    pip install loom-engine[hub]\n"
            "Loading a GGUF you already have with `loom.Model.from_file` needs nothing."
        ) from None
    return huggingface_hub


def download(
    repo_id: str,
    filename: str | None = None,
    revision: str = "main",
    cache_dir: str | os.PathLike | None = None,
    token: str | None = None,
) -> Path:
    """The local path of a GGUF in `repo_id`, downloading it if it is not already cached.

    **Choosing the file when the caller did not.** A repo publishing one model has one `.gguf` and
    naming it every time is noise. A repo publishing several -- quantisations, or a family -- has no
    obvious default, so this raises and lists them instead of picking. Guessing there means somebody
    eventually reports numbers for a quantisation they did not choose and cannot tell they got.
    """
    hub = _require_hub()

    if filename is None:
        listing = hub.list_repo_files(repo_id, revision=revision, token=token)
        candidates = sorted(name for name in listing if name.endswith(GGUF_SUFFIX))
        if not candidates:
            raise FileNotFoundError(f"{repo_id} (revision {revision}) contains no {GGUF_SUFFIX} file")
        if len(candidates) > 1:
            listed = "\n  ".join(candidates)
            raise ValueError(
                f"{repo_id} contains {len(candidates)} GGUF files; name the one you want with "
                f"`filename=`:\n  {listed}"
            )
        filename = candidates[0]

    return Path(hub.hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None, token=token,
    ))
