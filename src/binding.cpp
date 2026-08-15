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
#include "loom/core/model_contract.h"
#include "loom/core/text_generate.h"
#include "loom/core/transcribe.h"

#include <ggml.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

// The model's own tokenizer, if it carries one.
//
// **Five vocabulary families and one dispatch, taken from `tools/loom_cli/main.cpp`** -- the only
// other host that does this, and the one that learned the ordering the hard way. A GGUF says which it
// has in `tokenizer.ggml.model`: "gpt2" is byte-level BPE, "bert" is WordPiece, "byt5" is byte-level,
// "llama"/"t5" are SentencePiece, and "supertonic" is SupertonicTTS's grapheme codepoint table. The
// five classes share no base, so this holds one of each and asks whichever is non-null.
//
// **Every family is dispatched by NAME, and an unrecognized tag yields no tokenizer rather than an
// error.** This used to be four `if`s and an `else` that fell through to `Vocab::load`, which throws on
// any tag that is not "llama"/"t5" (loom.cpp `src/core/vocab.cpp`) -- and that throw happened inside
// `Model`'s constructor, so a GGUF carrying a vocabulary this binding had not been taught yet did not
// merely lack a tokenizer, it FAILED TO LOAD AT ALL. Adding "supertonic" to the exporter's output would
// have broken loading every Supertonic model, which is how this was found. A name-per-family cascade
// keeps `Vocab::load` seeing only the two tags it accepts, and makes the next new family a missing
// feature instead of a broken file.
//
// **A model may carry none, and that is not a defect.** The phoneme-input TTS families (Matcha, VITS,
// Kokoro, StyleTTS2) consume ids a phonemiser produces outside the engine, so their GGUFs have no
// vocabulary to embed. `kind()` is empty for those and `Model.tokenizer` is None on the Python side,
// which is a better answer than an object whose every method raises.
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
        } else if (tokenizer->kind_ == "supertonic") {
            tokenizer->supertonic_ = loom::SupertonicTextVectorizer::load(model);
        } else if (tokenizer->kind_ == "llama" || tokenizer->kind_ == "t5") {
            tokenizer->spm_ = loom::Vocab::load(model);
        }
        if (!tokenizer->valid()) return nullptr;
        return tokenizer;
    }

    bool valid() const { return spm_ || bpe_ || wordpiece_ || byte_ || supertonic_; }
    const std::string& kind() const { return kind_; }

    size_t size() const {
        if (spm_) return spm_->size();
        if (bpe_) return bpe_->size();
        if (wordpiece_) return wordpiece_->size();
        if (byte_) return byte_->size();
        // n_tokens(), not vocab_size() -- the latter is the 65536-entry BMP lookup table, which is not
        // what any caller reading `tokenizer.size()` means.
        return supertonic_->n_tokens();
    }

    // `lang` is the "optional argument, else the model's own declared default" shape: only a vocabulary
    // that is parameterized by language can honour it, and the rest say so rather than ignoring it. An
    // argument silently dropped is the failure mode this rejection exists to prevent -- a caller asking
    // for Korean and quietly getting English.
    std::vector<int32_t> encode(const std::string& text, const std::string& lang) const {
        if (!lang.empty() && !supertonic_) {
            throw loom::SchemaError("tokenizer kind '" + kind_ + "' takes no language argument; only a "
                                    "vocabulary that tags its input by language (supertonic) does");
        }
        if (spm_) return spm_->encode(text);
        if (bpe_) return bpe_->encode(text);
        if (wordpiece_) return wordpiece_->encode(text);
        if (byte_) return byte_->encode(text);
        return supertonic_->tokenize(text, lang);  // empty lang -> the file's own default_lang
    }

    std::string decode(const std::vector<int32_t>& ids) const {
        if (spm_) return spm_->decode(ids);
        if (bpe_) return bpe_->decode(ids);
        if (wordpiece_) return wordpiece_->decode(ids);
        if (byte_) return byte_->decode(ids);
        return supertonic_->detokenize(ids);
    }

    // Empty for every family that has no such concept, which is every family but supertonic.
    std::string default_lang() const { return supertonic_ ? supertonic_->default_lang() : std::string{}; }

private:
    Tokenizer() = default;
    std::string kind_;
    std::unique_ptr<loom::Vocab> spm_;
    std::unique_ptr<loom::BpeVocab> bpe_;
    std::unique_ptr<loom::WordPieceVocab> wordpiece_;
    std::unique_ptr<loom::ByteVocab> byte_;
    std::unique_ptr<loom::SupertonicTextVectorizer> supertonic_;
};

// One driver input, marshalled. A driver's world is numbers and arrays of numbers -- the bridge's own
// Value variant is exactly those two -- so this is the whole conversion, and it is a free function
// because two entry points need it: `call`, and the extra inputs a `generate` forwards.
loom::LoomLuaBridge::Value to_value(const std::string& key, const py::handle& value) {
    if (py::isinstance<py::float_>(value) || py::isinstance<py::int_>(value)) {
        return value.cast<double>();
    }
    try {
        return value.cast<std::vector<double>>();
    } catch (const py::cast_error&) {
        throw std::runtime_error(
            "input '" + key + "' is neither a number nor a sequence of numbers. A driver's "
            "inputs are numbers and arrays of numbers; anything with more structure than "
            "that belongs in the model, not in the call.");
    }
}

// Owns everything the engine needs alive for the duration of a session, in an order that matters:
// the bridge holds non-owning references to the model and the cache, so they are declared first and
// destroyed last.
class Model {
public:
    // `device` is a loom device spec -- "" (which defers to $LOOM_DEVICE, then autodetection), "auto",
    // "cpu", "gpu", or a device name like "Vulkan0". The default is empty rather than "cpu" so that a
    // wheel built with a GPU backend uses it without every caller having to ask, while a wheel built
    // without one resolves to exactly the CPU it always did.
    explicit Model(const std::string& path, const std::string& device = std::string{})
        : device_(std::make_unique<loom::Device>(loom::Device::open(device))) {
        const loom::Backends backends = device_->backends();
        model_ = loom::GgufModel::load(path, backends);
        if (model_ == nullptr) throw std::runtime_error("could not load a loom GGUF from " + path);

        bridge_ = std::make_unique<loom::LoomLuaBridge>(backends);
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
                kv_cache_ = loom::make_kv_cache(*model_, device_->backends());
            }
            if (topo.uses_conv_state() && conv_state_ == nullptr) {
                conv_state_ = loom::make_conv_state_cache(*model_, device_->backends());
            }
            loom::KvCache* kv_for_module = topo.uses_kv_cache() ? kv_cache_.get() : nullptr;
            loom::ConvStateCache* conv_for_module = topo.uses_conv_state() ? conv_state_.get() : nullptr;
            bridge_->register_module(name, *model_, std::move(topo), kv_for_module, conv_for_module);
        }

        // Guarded, not bare: `kv_str` THROWS on a missing key, so an unconditional read here made a
        // GGUF without a driver fail to construct at all -- even though `has_driver()` below is written
        // for exactly that state and `call()` already refuses with a real message when it is empty. Same
        // shape of bug as the tokenizer dispatch above, in the same constructor: an optional property of
        // the file read as if it were mandatory.
        if (model_->has_kv("model.driver_script")) driver_ = model_->kv_str("model.driver_script");
        if (!driver_.empty()) bridge_->load_script(driver_);

        tokenizer_ = Tokenizer::load(*model_);
    }

    // What this session actually resolved to -- the ggml device name and its human-readable
    // description. Worth exposing because "device=''" means "decide for me", and a caller who did not
    // choose still needs to be able to find out what was chosen.
    std::string device_name() const { return device_->name(); }
    std::string device_description() const { return device_->description(); }

    bool has_tokenizer() const { return tokenizer_ != nullptr; }
    std::string tokenizer_kind() const { return tokenizer_ ? tokenizer_->kind() : std::string{}; }
    size_t tokenizer_size() const { return require_tokenizer().size(); }
    std::string tokenizer_default_lang() const {
        return tokenizer_ ? tokenizer_->default_lang() : std::string{};
    }
    std::vector<int32_t> encode(const std::string& text, const std::string& lang) const {
        return require_tokenizer().encode(text, lang);
    }
    std::string decode(const std::vector<int32_t>& ids) const { return require_tokenizer().decode(ids); }

    std::vector<std::string> topologies() const { return names_; }
    std::string architecture() const { return model_->architecture(); }

    // WHAT THIS FILE SAYS IT IS, which is what the Python layer dispatches its end-to-end doors on.
    // `architecture` above is a per-MODEL name, so anything keyed on it would be a table of model names
    // living in this package -- exactly what loom-py's CLAUDE.md forbids and what the declared contract
    // exists to replace (loom.cpp docs/HIGH-LEVEL-API.md).
    //
    // Marshalled as a dict rather than a bound struct: every field is optional and absence is
    // meaningful, and a dict says "this key was not declared" in the one way Python already reads
    // without a sentinel per field. `declared` is separate because a host must be able to tell a file
    // that states its contract from one it has to be told about.
    py::dict contract() const {
        const loom::ModelContract c = loom::ModelContract::read(*model_);
        py::dict out;
        out["declared"] = c.declared();
        out["task"] = c.task;
        out["input_kind"] = c.input_kind;
        out["output_kind"] = c.output_kind;
        out["interface"] = c.interface_name();
        out["sample_rate"] = c.sample_rate;
        out["clip_samples"] = c.clip_samples;
        out["max_input_tokens"] = c.max_input_tokens;
        out["text_frontend"] = c.text_frontend;
        out["phoneme_alphabet"] = c.phoneme_alphabet;
        out["phonemizer_ruleset"] = c.phonemizer_ruleset;
        out["languages"] = c.languages;
        out["entry_points"] = c.entry_points;
        out["default_steps"] = c.default_steps;
        out["voices"] = c.voices;
        return out;
    }

    // The causal-LM decode loop, which is the ENGINE's (loom/core/text_generate.h) rather than a Python
    // reimplementation of it. The Python one that used to live in `Model.generate_ids` was correct and
    // was still a second copy: loom_cli's differed in three ways, and nothing but coincidence kept this
    // one in step. Both hosts call the same function now.
    std::vector<int32_t> generate(const std::vector<int32_t>& tokens, uint32_t max_new_tokens,
                                  int32_t eos_token, const py::dict& extra_inputs) {
        if (driver_.empty()) {
            throw std::runtime_error(
                "this GGUF carries no driver script, so there is nothing to generate with.");
        }
        loom::text::GenerateOptions options;
        options.max_new_tokens = max_new_tokens;
        options.eos_token = eos_token;
        // Anything else the caller named, forwarded verbatim to the driver -- a model whose `infer`
        // takes more than tokens (a style vector, a speaker id) is driven through the same loop.
        for (auto item : extra_inputs) {
            const auto key = item.first.cast<std::string>();
            options.extra_inputs[key] = to_value(key, item.second);
        }
        return loom::text::generate(*bridge_, *model_, tokens, options);
    }
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
    // TRANSCRIPTION, which is the engine's `loom::audio::transcribe` and nothing else (see
    // loom/core/transcribe.h). An earlier draft of this method reimplemented the windowing here and
    // accepted that Python would transcribe long audio worse than the CLI, because the timestamp-aware
    // seek "belonged" to the CLI. It did not: the timestamp ids come from the vocabulary the GGUF
    // embeds and the frame duration is arithmetic on declared hparams, so the engine can do it for
    // everyone. Both front ends now run the identical loop -- windowing, segment splitting, seeking on
    // the model's own boundaries, prev_tokens conditioning.
    //
    // Returns the segments, because throwing them away here would be inventing the same asymmetry in a
    // different place: a caller who wants one string joins them, and a caller building subtitles or
    // seeking in a player needs the times. `Model.transcribe` in loom/__init__.py is what turns this
    // into either shape.
    py::object transcribe(const std::vector<float>& waveform, const py::dict& options) {
        if (driver_.empty()) {
            throw std::runtime_error(
                "this GGUF carries no driver script, so there is nothing to transcribe.");
        }
        loom::audio::TranscribeOptions opts;
        // Names, resolved by the engine against the file's own vocabulary -- a Python caller has no
        // way to look up the id of `<|en|>`, and used to have to.
        if (options.contains("language")) opts.language = options["language"].cast<std::string>();
        if (options.contains("task")) opts.task = options["task"].cast<std::string>();
        if (options.contains("timestamps")) opts.timestamps = options["timestamps"].cast<bool>();
        if (options.contains("condition_on_previous")) {
            opts.condition_on_previous = options["condition_on_previous"].cast<bool>();
        }

        const loom::audio::Transcription result =
            loom::audio::transcribe(*bridge_, *model_, waveform, opts);

        py::list segments;
        for (const loom::audio::Segment& seg : result.segments) {
            py::dict d;
            d["start"] = seg.start;
            d["end"] = seg.end;
            d["text"] = seg.text;
            d["closed"] = seg.closed;
            segments.append(d);
        }
        py::dict out;
        out["segments"] = segments;
        out["text"] = result.text;
        out["windows"] = result.windows;
        out["timestamped"] = result.timestamped;
        return out;
    }

    py::object call(const std::string& fn_name, const py::dict& inputs) {
        if (driver_.empty()) {
            throw std::runtime_error(
                "this GGUF carries no driver script, so there is nothing to call. Its topologies can "
                "still be listed and built, but running it is the host's job.");
        }
        std::unordered_map<std::string, loom::LoomLuaBridge::Value> args;
        for (auto item : inputs) {
            args.emplace(item.first.cast<std::string>(), to_value(item.first.cast<std::string>(),
                                                                   item.second));
        }
        // Held in a named local before converting: `call` returns by value, and binding a reference
        // into the returned variant reads freed memory -- the bug that cost a full bisect on the C++
        // side, and there is no reason to rediscover it here.
        const loom::LoomLuaBridge::Value result = bridge_->call(fn_name, args);
        if (std::holds_alternative<double>(result)) return py::float_(std::get<double>(result));
        return py::cast(std::get<std::vector<double>>(result));
    }

private:
    // Declared first so it outlives everything holding its backend handles -- the model's weight
    // buffer, the caches and every graph the bridge builds. Held by pointer only because loom::Device
    // has no default constructor and this member is initialized in the constructor's body order.
    std::unique_ptr<loom::Device> device_;
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
                "this GGUF carries no tokenizer vocabulary. Its driver takes ids directly -- the "
                "phoneme-input TTS families (Matcha, VITS, Kokoro, StyleTTS2) consume ids a phonemiser "
                "produces outside the engine, so there is nothing here to encode text with.");
        }
        return *tokenizer_;
    }
};

} // namespace

namespace {

// ggml logs the loading of every backend at INFO, so a GGML_BACKEND_DL build prints a line like
//   load_backend: loaded CPU backend from .../loom/libggml-cpu-haswell.so
// to stderr on `import loom`. A statically linked build never did -- there was nothing to load -- so
// this is noise the packaging change introduced rather than something the engine always did, and a
// library that writes to stderr when merely imported is a library that shows up in somebody else's
// clean output.
//
// Dropped rather than silenced wholesale: WARN and ERROR still go to stderr, because those are the
// messages that explain a backend which loaded and then found no usable device -- the exact failure
// `loom.devices()` exists to make visible, and one that would be much worse to hide.
//
// GGML_LOG_LEVEL_CONT means "a continuation of the previous message", so it has no level of its own
// and has to inherit the decision made for what it continues; otherwise a dropped INFO can be followed
// by a printed fragment of itself.
void forward_ggml_log(ggml_log_level level, const char* text, void* /*user_data*/) {
    static bool last_was_printed = false;
    if (level != GGML_LOG_LEVEL_CONT) {
        last_was_printed = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR;
    }
    if (last_was_printed && text != nullptr) {
        std::fputs(text, stderr);
    }
}

} // namespace

PYBIND11_MODULE(_loom, m) {
    m.doc() = "Low-level bindings to loom.cpp. The API you want is in `loom`, not here.";

    // Before anything can load a backend -- which, in this build, is the first thing that happens.
    ggml_log_set(forward_ggml_log, nullptr);

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

    // Where ggml looks for backend .so files. Its own default search is the executable's directory and
    // the current directory, and inside an interpreter the executable is `python` -- so a wheel that
    // did not call this would find no backends at all, the CPU included (BACKLOG.md P4.8). Called from
    // loom/__init__.py with this package's directory and every installed `loom_rt_*` accelerator
    // package, which is why that discovery lives in Python where it can use importlib.
    m.def("add_backend_search_path", &loom::add_backend_search_path, py::arg("directory"),
          "Add a directory to search for dynamically loaded ggml backends.");

    // What actually got loaded. The point of exposing it is that a wheel's accelerator is now an
    // install-time question: `loom.devices()` is how a caller checks whether `loom-py-rt-vulkan` did
    // anything, without loading a model to find out.
    m.def("devices", [] {
        py::list out;
        for (const loom::DeviceInfo& dev : loom::available_devices()) {
            py::dict entry;
            entry["name"] = dev.name;
            entry["description"] = dev.description;
            entry["is_cpu"] = dev.is_cpu;
            entry["memory_free"] = dev.memory_free;
            entry["memory_total"] = dev.memory_total;
            out.append(std::move(entry));
        }
        return out;
    }, "Every ggml device visible to this process, in registration order.");

    py::class_<Model>(m, "Model")
        .def(py::init<const std::string&, const std::string&>(), py::arg("path"),
              py::arg("device") = std::string{})
        .def("device_name", &Model::device_name)
        .def("device_description", &Model::device_description)
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
        .def("tokenizer_default_lang", &Model::tokenizer_default_lang)
        .def("encode", &Model::encode, py::arg("text"), py::arg("lang") = std::string{})
        .def("decode", &Model::decode, py::arg("ids"))
        .def("call", &Model::call, py::arg("fn_name"), py::arg("inputs"))
        .def("transcribe", &Model::transcribe, py::arg("waveform"), py::arg("options"))
        .def("contract", &Model::contract)
        .def("generate", &Model::generate, py::arg("tokens"), py::arg("max_new_tokens"),
             py::arg("eos_token"), py::arg("extra_inputs"));
}
