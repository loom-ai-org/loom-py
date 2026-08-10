// The pybind11 layer: one class, because a loom model is one thing.
//
// **What a binding for this engine has to expose is unusually small**, and that is the architecture
// showing through rather than an unfinished job. A loom GGUF carries its own graph topologies and its
// own driver script, so there is no per-architecture Python to write: loading a model means
// registering every topology it declares and handing the engine the Lua it shipped with, and running
// one means calling that driver's entry point. A model this layer has never heard of works the day
// the exporter can produce it.
//
// So the C++ here is deliberately the same sequence every one of the engine's own end-to-end tests
// performs by hand -- register modules, attach a cache to the ones that declare they need it, load
// the script, call `infer` -- with the per-test hardcoding replaced by `topology_names()` and
// `GraphTopology::uses_kv_cache()`. Everything else, including how a host is supposed to shape its
// inputs, belongs in `loom/__init__.py` where it can be read and changed without a compiler.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

// Owns everything the engine needs alive for the duration of a session, in an order that matters:
// the bridge holds non-owning references to the model and the cache, so they are declared first and
// destroyed last.
class Model {
public:
    explicit Model(const std::string& path)
        : backend_(ggml_backend_cpu_init()) {
        if (backend_ == nullptr) throw std::runtime_error("could not initialise the CPU backend");
        model_ = loom::GgufModel::load(path, backend_.get());
        if (model_ == nullptr) throw std::runtime_error("could not load a loom GGUF from " + path);

        bridge_ = std::make_unique<loom::LoomLuaBridge>(backend_.get());
        names_ = model_->topology_names();
        // A cache is made only if some topology says it wants one. `make_kv_cache` reads the model's
        // own declared geometry, which is why no caller ever passes a context length here.
        bool wants_cache = false;
        for (const std::string& name : names_) {
            if (loom::GraphTopology::parse(model_->topology_json(name)).uses_kv_cache()) wants_cache = true;
        }
        if (wants_cache) kv_cache_ = loom::make_kv_cache(*model_, backend_.get());

        for (const std::string& name : names_) {
            loom::GraphTopology topo = loom::GraphTopology::parse(model_->topology_json(name));
            const bool cached = topo.uses_kv_cache();
            bridge_->register_module(name, *model_, std::move(topo),
                                     cached ? kv_cache_.get() : nullptr);
        }

        driver_ = model_->kv_str("model.driver_script");
        if (!driver_.empty()) bridge_->load_script(driver_);
    }

    std::vector<std::string> topologies() const { return names_; }
    std::string architecture() const { return model_->architecture(); }
    bool has_driver() const { return !driver_.empty(); }
    std::string driver_source() const { return driver_; }

    uint32_t hparam_u32(const std::string& key) const { return model_->hparam_u32(key); }
    float hparam_f32(const std::string& key) const { return model_->hparam_f32(key); }
    std::string hparam_str(const std::string& key) const { return model_->hparam_str(key); }

    // `inputs` is {name: float | sequence[float]}, which is exactly the bridge's own Value variant --
    // a driver's world is numbers and arrays of numbers, and nothing here needs to know that one
    // model's array is a waveform and another's is a run of token ids.
    py::object call(const std::string& fn_name, const py::dict& inputs) {
        if (driver_.empty()) {
            throw std::runtime_error(
                "this GGUF carries no driver script, so there is nothing to call. Its topologies can "
                "still be listed and built, but running it is the host's job.");
        }
        std::unordered_map<std::string, loom::LoomLuaBridge::Value> args;
        for (auto item : inputs) {
            const auto key = item.first.cast<std::string>();
            const py::handle value = item.second;
            if (py::isinstance<py::float_>(value) || py::isinstance<py::int_>(value)) {
                args.emplace(key, value.cast<double>());
            } else {
                try {
                    args.emplace(key, value.cast<std::vector<double>>());
                } catch (const py::cast_error&) {
                    throw std::runtime_error(
                        "input '" + key + "' is neither a number nor a sequence of numbers. A driver's "
                        "inputs are numbers and arrays of numbers; anything with more structure than "
                        "that belongs in the model, not in the call.");
                }
            }
        }
        // Held in a named local before converting: `call` returns by value, and binding a reference
        // into the returned variant reads freed memory -- the bug that cost a full bisect on the C++
        // side, and there is no reason to rediscover it here.
        const loom::LoomLuaBridge::Value result = bridge_->call(fn_name, args);
        if (std::holds_alternative<double>(result)) return py::float_(std::get<double>(result));
        return py::cast(std::get<std::vector<double>>(result));
    }

private:
    ggml_backend_ptr backend_;
    std::unique_ptr<loom::GgufModel> model_;
    std::unique_ptr<loom::KvCache> kv_cache_;
    std::unique_ptr<loom::LoomLuaBridge> bridge_;
    std::vector<std::string> names_;
    std::string driver_;
};

} // namespace

PYBIND11_MODULE(_loom, m) {
    m.doc() = "Low-level bindings to loom.cpp. The API you want is in `loom`, not here.";

    // loom::Error is what the engine raises for every bad-model / bad-input condition; letting it
    // reach Python as a plain RuntimeError would lose the distinction between "your GGUF is wrong"
    // and "this binding is wrong".
    static py::exception<loom::Error> loom_error(m, "LoomError", PyExc_RuntimeError);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const loom::Error& e) {
            py::set_error(loom_error, e.what());
        }
    });

    py::class_<Model>(m, "Model")
        .def(py::init<const std::string&>(), py::arg("path"))
        .def("topologies", &Model::topologies)
        .def("architecture", &Model::architecture)
        .def("has_driver", &Model::has_driver)
        .def("driver_source", &Model::driver_source)
        .def("hparam_u32", &Model::hparam_u32, py::arg("key"))
        .def("hparam_f32", &Model::hparam_f32, py::arg("key"))
        .def("hparam_str", &Model::hparam_str, py::arg("key"))
        .def("call", &Model::call, py::arg("fn_name"), py::arg("inputs"));
}
