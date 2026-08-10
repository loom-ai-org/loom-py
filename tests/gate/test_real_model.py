"""The binding against a real exported model.

Everything the CI suite checks is about the Python layer's own decisions. What it cannot check is the
claim this package actually makes: that loading a loom GGUF registers whatever topologies it declares,
attaches a cache to the ones that ask for it, and runs the driver the file shipped with -- for a model
this package has never heard of.

So this needs a real artifact, and skips without one:

    export LOOM_TEST_MODEL=~/loom-fixtures/matcha_mil.gguf
    pytest tests/gate -q

Deliberately written against no particular architecture. It asserts the *shape* of what a loom model
is, which is the whole of what this package knows -- a test that expected Matcha's inputs would be
this package learning about a model, which is the thing the design is supposed to make unnecessary.
"""
import os
from pathlib import Path

import pytest

import loom


@pytest.fixture(scope="module")
def model():
    path = os.environ.get("LOOM_TEST_MODEL")
    if not path:
        pytest.skip("LOOM_TEST_MODEL is not set; it names any loom GGUF to drive")
    if not Path(path).is_file():
        pytest.skip(f"{path} does not exist")
    return loom.Model.from_file(path)


def test_a_model_describes_itself(model):
    """The three things every loom GGUF carries, and the reason no per-architecture code is needed."""
    assert model.architecture
    assert model.topologies, "a loom GGUF declares at least one graph topology"
    assert all(isinstance(name, str) and name for name in model.topologies)
    assert isinstance(model.driver_source, str)
    assert model.has_driver == bool(model.driver_source)


def test_the_repr_is_useful_at_a_prompt(model):
    text = repr(model)
    assert model.architecture in text
    assert model.path.name in text


def test_a_wrong_hparam_kind_raises_rather_than_guessing(model):
    with pytest.raises(ValueError, match="unknown hparam kind"):
        model.hparam("anything", kind="int64")


def test_calling_a_function_the_driver_does_not_define(model):
    """The engine's own error type, surfaced as itself."""
    if not model.has_driver:
        pytest.skip("this artifact carries no driver")
    with pytest.raises(loom.LoomError, match="no such Lua function"):
        model.call("not_a_real_entry_point", {})


def test_inference_runs_and_is_deterministic(model):
    """`LOOM_TEST_MODEL_INPUTS` supplies the driver's own arguments as JSON, because which arguments a
    driver takes is a property of the model. Without it this test cannot know what to pass and says
    so, rather than guessing a signature and reporting the failure as a bug in the binding."""
    import json

    raw = os.environ.get("LOOM_TEST_MODEL_INPUTS")
    if not raw:
        pytest.skip("set LOOM_TEST_MODEL_INPUTS to a JSON object of the driver's inputs "
                    "(model.driver_source documents them)")
    inputs = json.loads(raw)

    first = model.infer(**inputs)
    assert isinstance(first, (list, float))
    if isinstance(first, list):
        assert first, "the driver returned an empty result"
        assert all(isinstance(x, float) for x in first)

    # A loom driver is deterministic given its inputs -- any randomness is seeded through them -- so
    # two identical calls must agree. This also exercises the second call against a cache and graph
    # buffers the first one left behind, which is where a stateful engine gets caught.
    assert model.infer(**inputs) == first
