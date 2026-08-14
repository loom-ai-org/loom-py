# loom-py-rt-vulkan

The Vulkan backend for [`loom-py-rt`](https://pypi.org/project/loom-py-rt/). One shared library in one
small wheel; there is nothing here to import.

```sh
pip install "loom-py-rt[vulkan]"
```

```python
import loom

loom.devices()                                   # a Vulkan device now appears
model = loom.Model.from_file(path, device="gpu")
```

The base package is unchanged by installing this — it discovers the library on `sys.path` at import
and `device="auto"` starts using it. That is what `GGML_BACKEND_DL` buys: no second copy of the
runtime per accelerator.

## If it appears not to have worked

Check `loom.devices()`. A backend whose driver is too old, or which finds no supported device, loads
**without error** and registers nothing — the only other symptom is a model running at CPU speed.

`device="gpu"` asks for an offload device with its own memory, preferring one the kernel confirms is a
GPU. It is not a promise that Vulkan specifically was chosen. Pass `device="Vulkan0"` to require this
backend.

## Versioning

This package pins the base with `==`, not `~=`. It carries a `libggml-vulkan.so` that is dlopened
beside the base wheel's `libggml-base.so`, and ggml makes no ABI promise across revisions — so a base
release that moves its ggml pin invalidates every backend wheel published before it. The exact pin is
what stops pip from pairing two libraries that do not agree.
