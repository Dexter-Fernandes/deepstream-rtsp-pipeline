/*
 * build_yolo26_engine — CLI tool to build a TRT engine from a YOLO26n ONNX
 * with the yolo26_decode plugin appended as the final layer.
 *
 * Usage:
 *   build_yolo26_engine <onnx> <plugin_lib> <engine_out> [--fp16] [--max-batch N]
 *
 * The plugin lib is dlopen'd so Yolo26DecodePluginCreator is registered in
 * the TRT plugin registry before the builder runs.
 */

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <dlfcn.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

// ---------------------------------------------------------------------------
// Minimal TRT logger
// ---------------------------------------------------------------------------

class Logger : public ILogger
{
    void log(Severity sev, const char* msg) noexcept override
    {
        if (sev <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
} gLogger;

// ---------------------------------------------------------------------------
// RAII helpers
// TRT 10+ removed destroy() — objects are deleted with operator delete.
// ---------------------------------------------------------------------------

struct TrtDelete { template<class T> void operator()(T* p) const { delete p; } };
template<class T> using TrtPtr = std::unique_ptr<T, TrtDelete>;

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <onnx> <plugin_lib> <engine_out> [--fp16] [--max-batch N]\n";
        return 1;
    }

    const std::string onnx_path   = argv[1];
    const std::string plugin_lib  = argv[2];
    const std::string engine_out  = argv[3];
    bool              fp16        = false;
    int               max_batch   = 3;

    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fp16") == 0)
            fp16 = true;
        else if (std::strcmp(argv[i], "--max-batch") == 0 && i + 1 < argc)
            max_batch = std::stoi(argv[++i]);
    }

    // Load plugin .so — registers Yolo26DecodePluginCreator in TRT registry
    if (!dlopen(plugin_lib.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
        std::cerr << "dlopen failed for " << plugin_lib << ": " << dlerror() << "\n";
        return 1;
    }

    // Read ONNX bytes
    std::ifstream onnx_file(onnx_path, std::ios::binary);
    if (!onnx_file) { std::cerr << "Cannot open " << onnx_path << "\n"; return 1; }
    const std::string onnx_bytes(
        (std::istreambuf_iterator<char>(onnx_file)),
         std::istreambuf_iterator<char>());

    // Builder + network
    TrtPtr<IBuilder> builder(createInferBuilder(gLogger));
    // kEXPLICIT_BATCH = 0 in TRT 10 (deprecated enum, still valid value).
    TrtPtr<INetworkDefinition> network(builder->createNetworkV2(0U));
    TrtPtr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, gLogger));

    if (!parser->parse(onnx_bytes.data(), onnx_bytes.size())) {
        std::cerr << "ONNX parse failed\n";
        for (int i = 0; i < parser->getNbErrors(); ++i)
            std::cerr << "  " << parser->getError(i)->desc() << "\n";
        return 1;
    }

    // Unmark YOLO26n output, append decode plugin
    ITensor* yolo_out = network->getOutput(0);
    network->unmarkOutput(*yolo_out);

    IPluginRegistry* registry = getPluginRegistry();
    IPluginCreator*  creator  = registry->getPluginCreator("Yolo26DecodePlugin", "1", "");
    if (!creator) {
        std::cerr << "Yolo26DecodePlugin not found in registry\n"; return 1;
    }
    PluginFieldCollection empty_fields{};
    TrtPtr<IPluginV2> plugin(creator->createPlugin("yolo26_decode", &empty_fields));
    ITensor* inputs[] = { yolo_out };
    ILayer* decode_layer = network->addPluginV2(inputs, 1, *plugin);
    decode_layer->setName("yolo26_decode");
    network->markOutput(*decode_layer->getOutput(0));

    // Builder config
    TrtPtr<IBuilderConfig> cfg(builder->createBuilderConfig());
    cfg->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);
    if (fp16)
        cfg->setFlag(BuilderFlag::kFP16);  // kFP16 = 0, deprecated name but still valid

    // Optimization profile for dynamic batch
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("images", OptProfileSelector::kMIN, Dims4{1, 3, 640, 640});
    profile->setDimensions("images", OptProfileSelector::kOPT, Dims4{max_batch, 3, 640, 640});
    profile->setDimensions("images", OptProfileSelector::kMAX, Dims4{max_batch, 3, 640, 640});
    cfg->addOptimizationProfile(profile);

    std::cerr << "[build_engine] Building engine (fp16=" << fp16
              << ", max_batch=" << max_batch << ")…\n";

    TrtPtr<IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *cfg));
    if (!serialized) {
        std::cerr << "Engine build failed\n"; return 1;
    }

    std::ofstream out(engine_out, std::ios::binary);
    if (!out) { std::cerr << "Cannot write " << engine_out << "\n"; return 1; }
    out.write(static_cast<const char*>(serialized->data()), serialized->size());
    std::cerr << "[build_engine] Engine saved → " << engine_out << "\n";
    return 0;
}
