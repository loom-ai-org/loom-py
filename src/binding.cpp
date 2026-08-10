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

// The model's own tokenizer, if it carries one.
//
// **Four vocabulary families and one dispatch, taken from `tools/loom_cli/main.cpp`** -- the only
// other host that does this, and the one that learned the ordering the hard way. A GGUF says which it
// has in `tokenizer.ggml.model`: "gpt2" is byte-level BPE, "bert" is WordPiece, "byt5" is byte-level,
// and anything else ("llama", "t5") is SentencePiece. The four classes share no base, so this holds
// one of each and asks whichever is non-null.
//
// **A model may carry none, and that is not a defect.** The TTS families consume phoneme ids that a
// phonemiser produces outside the engine, so their GGUFs have no vocabulary to embed. `kind()` is
// empty for those and `Model.tokenizer` is None on the Python side, which is a better answer than an
// object whose every method raises.
class Tokenizer {
public:
    static std::unique_ptr<Tokenizer> load(const loom::GgufModel& model) {
        if (!model.has_kv("tokenizer.ggml.model")) return nullptr;
        std::unique_ptr<Tokenizer> tokenizer(new Tokenizer());
        tokenizer->kind_ = model.kv_str("tokenizer.ggml.model");
        if (tokenizer->kind_ == "bert") {
            tokenizer->wordpiece_ = loom::WordPieceVocab::load(model);
        } else if (tokenizer->kind_ == "byt5") {
            tokenizer->byte_ = loom::ByteVocab::load(model);
        } else if (tokenizer->kind_ == "gpt2") {
            tokenizer->bpe_ = loom::BpeVocab::load(model);
        } else {
            // SentencePiece. `Vocab::load` THROWS on a gpt2 file where `BpeVocab::load` merely returns
            // null, which is why gpt2 is dispatched by name rather than probed for -- the ordering
            // `loom_cli` records.
            tokenizer->spm_ = loom::Vocab::load(model);
        }
        if (!tokenizer->valid()) return nullptr;
        return tokenizer;
    }

    bool valid() const { return spm_ || bpe_ || wordpiece_ || byte_; }
    const std::string& kind() const { return kind_; }

    size_t size() const {
        if (spm_) return spm_->size();
        if (bpe_) return bpe_->size();
        if (wordpiece_) return wordpiece_->size();
        return byte_->size();
    }

    std::vector<int32_t> encode(const std::string& text) const {
        if (spm_) return spm_->encode(text);
        if (bpe_) return bpe_->encode(text);
        if (wordpiece_) return wordpiece_->encode(text);
        return byte_->encode(text);
    }

    std::string decode(const std::vector<int32_t>& ids) const {
        if (spm_) return spm_->decode(ids);
        if (bpe_) return bpe_->decode(ids);
        if (wordpiece_) return wordpiece_->decode(ids);
        return byte_->decode(ids);
    }

private:
    Tokenizer() = default;
    std::string kind_;
    std::unique_ptr<loom::Vocab> spm_;
    std::unique_ptr<loom::BpeVocab> bpe_;
    std::unique_ptr<loom::WordPieceVocab> wordpiece_;
    std::unique_ptr<loom::ByteVocab> byte_;
};

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

        // A cache is made only if some topology says it wants one, and there are TWO kinds. Attention
        // blocks want a `KvCache`; a hybrid's ShortConv blocks carry their own history, which the KV
        // cache does not hold, and want a `ConvStateCache` (BACKLOG.md P4.0.10). LFM2 is the model
        // that has both, and a binding that made only the first loaded it, tokenized for it, and then
        // failed inside the driver on the eleventh node.
        //
        // Both are sized from the file's own declared geometry, which is why no caller passes a
        // context length: the model states it, `make_*_cache` reads it.
        for (const std::string& name : names_) {
            loom::GraphTopology topo = loom::GraphTopology::parse(model_->topology_json(name));
            if (topo.uses_kv_cache() && kv_cache_ == nullptr) {
                kv_cache_ = loom::make_kv_cache(*model_, backend_.get());
            }
            if (topo.uses_conv_state() && conv_state_ == nullptr) {
                conv_state_ = loom::make_conv_state_cache(*model_, backend_.get());
            }
            loom::KvCache* kv_for_module = topo.uses_kv_cache() ? kv_cache_.get() : nullptr;
            loom::ConvStateCache* conv_for_module = topo.uses_conv_state() ? conv_state_.get() : nullptr;
            bridge_->register_module(name, *model_, std::move(topo), kv_for_module, conv_for_module);
        }

        driver_ = model_->kv_str("model.driver_script");
        if (!driver_.empty()) bridge_->load_script(driver_);

        tokenizer_ = Tokenizer::load(*model_);
    }

    bool has_tokenizer() const { return tokenizer_ != nullptr; }
    std::string tokenizer_kind() const { return tokenizer_ ? tokenizer_->kind() : std::string{}; }
    size_t tokenizer_size() const { return require_tokenizer().size(); }
    std::vector<int32_t> encode(const std::string& text) const { return require_tokenizer().encode(text); }
    std::string decode(const std::vector<int32_t>& ids) const { return require_tokenizer().decode(ids); }

    std::vector<std::string> topologies() const { return names_; }
    std::string architecture() const { return model_->architecture(); }
    bool has_driver() const { return !driver_.empty(); }
    std::string driver_source() const { return driver_; }

    uint32_t hparam_u32(const std::string& key) const { return model_->hparam_u32(key); }
    float hparam_f32(const std::string& key) const { return model_->hparam_f32(key); }
    std::string hparam_str(const std::string& key) const { return model_->hparam_str(key); }
    // Full-key, with a default: the eos id lives at `tokenizer.ggml.eos_token_id`, outside the `loom.`
    // namespace the hparam_* accessors prefix, and a model may not declare one at all.
    int32_t kv_i32(const std::string& key, int32_t fallback) const { return model_->kv_i32(key, fallback); }

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
    std::unique_ptr<loom::ConvStateCache> conv_state_;
    std::unique_ptr<loom::LoomLuaBridge> bridge_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::vector<std::string> names_;
    std::string driver_;

    const Tokenizer& require_tokenizer() const {
        if (tokenizer_ == nullptr) {
            throw std::runtime_error(
                "this GGUF carries no tokenizer vocabulary. Its driver takes ids directly -- the TTS "
                "families consume phoneme ids a phonemiser produces outside the engine, so there is "
                "nothing here to encode text with.");
        }
        return *tokenizer_;
    }
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
        .def("kv_i32", &Model::kv_i32, py::arg("key"), py::arg("fallback"))
        .def("has_tokenizer", &Model::has_tokenizer)
        .def("tokenizer_kind", &Model::tokenizer_kind)
        .def("tokenizer_size", &Model::tokenizer_size)
        .def("encode", &Model::encode, py::arg("text"))
        .def("decode", &Model::decode, py::arg("ids"))
        .def("call", &Model::call, py::arg("fn_name"), py::arg("inputs"));
}
