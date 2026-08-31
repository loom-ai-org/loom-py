# loom-py-rt-metal

The Metal backend for [`loom-py-rt`](https://pypi.org/project/loom-py-rt/). One shared library in one
small wheel; there is nothing here to import.

```sh
pip install "loom-py-rt[metal]"
```

```python
import loom

loom.devices()                                   # a Metal device now appears
model = loom.Model.from_file(path, device="gpu")
```

The base package is unchanged by installing this — it discovers the library on `sys.path` at import
and `device="auto"` starts using it. That is what `GGML_BACKEND_DL` buys: no second copy of the
runtime per accelerator.

## Apple Silicon only

There is no Intel-Mac wheel, so `pip install "loom-py-rt[metal]"` will not resolve on one. That is
intended rather than an oversight — an extra that quietly resolved to nothing would hand back a
successful install and no Metal, and the first sign of it would be an unexplained performance number
much later.

**Metal is not the Neural Engine.** Metal is the GPU, and is an ordinary in-tree ggml backend. The
Neural Engine means CoreML, which no ggml backend targets.

## If it appears not to have worked

Check `loom.devices()`. A backend that finds no supported device loads **without error** and
registers nothing — the only other symptom is a model running at CPU speed.

`device="gpu"` asks for an offload device with its own memory, preferring one the kernel confirms is
a GPU. It is not a promise that Metal specifically was chosen. Pass `device="Metal0"` to require this
backend.

Note that Apple Silicon is a **unified-memory** part: the CPU and the GPU address the same physical
RAM. A device that is faster for a large matrix multiplication is not automatically faster for a
small graph, because the win no longer includes escaping a slow bus.

## Versioning

This package pins the base with `==`, not `~=`. It carries a `libggml-metal.so` that is dlopened
beside the base wheel's `libggml-base.dylib`, and ggml makes no ABI promise across revisions — so a
base release that moves its ggml pin invalidates every backend wheel published before it. The exact
pin is what stops pip from pairing two libraries that do not agree.
