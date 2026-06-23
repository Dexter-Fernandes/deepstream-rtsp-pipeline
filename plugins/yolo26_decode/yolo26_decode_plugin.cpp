#include "yolo26_decode_plugin.hpp"
#include "yolo26_decode_kernel.cuh"

#include <cstdio>
#include <cstring>
#include <stdexcept>

static const char* PLUGIN_NAME    = "Yolo26DecodePlugin";
static const char* PLUGIN_VERSION = "1";

// ---------------------------------------------------------------------------
// Yolo26DecodePlugin
// ---------------------------------------------------------------------------

Yolo26DecodePlugin::Yolo26DecodePlugin(const void* /*data*/, size_t /*length*/)
{
    // No serialized attributes — nothing to deserialize.
}

const char* Yolo26DecodePlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* Yolo26DecodePlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

nvinfer1::IPluginV2DynamicExt* Yolo26DecodePlugin::clone() const noexcept
{
    return new Yolo26DecodePlugin();
}

nvinfer1::DataType Yolo26DecodePlugin::getOutputDataType(
    int /*index*/,
    const nvinfer1::DataType* /*inputTypes*/,
    int /*nbInputs*/) const noexcept
{
    return nvinfer1::DataType::kFLOAT;
}

// Output shape mirrors input shape: [batch, N, 6] → [batch, N, 6].
nvinfer1::DimsExprs Yolo26DecodePlugin::getOutputDimensions(
    int /*outputIndex*/,
    const nvinfer1::DimsExprs* inputs,
    int /*nbInputs*/,
    nvinfer1::IExprBuilder& /*exprBuilder*/) noexcept
{
    return inputs[0];
}

bool Yolo26DecodePlugin::supportsFormatCombination(
    int                               pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int                               /*nbInputs*/,
    int                               /*nbOutputs*/) noexcept
{
    return inOut[pos].type   == nvinfer1::DataType::kFLOAT
        && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

int Yolo26DecodePlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* /*outputDesc*/,
    const void* const* inputs,
    void* const*       outputs,
    void*              /*workspace*/,
    cudaStream_t       stream) noexcept
{
    // Input is [batch, n_dets, 6].  Process all (batch × n_dets) rows.
    const auto& dims  = inputDesc[0].dims;
    const int   batch  = dims.d[0];
    const int   n_dets = dims.d[1];

    const float* d_in  = static_cast<const float*>(inputs[0]);
    float*       d_out = static_cast<float*>(outputs[0]);

    yolo26_decode(d_in, d_out, batch * n_dets, stream);
    return 0;
}

// ---------------------------------------------------------------------------
// Yolo26DecodePluginCreator
// ---------------------------------------------------------------------------

Yolo26DecodePluginCreator::Yolo26DecodePluginCreator()
{
    mFC.nbFields = 0;
    mFC.fields   = nullptr;
    std::fprintf(stderr, "[Yolo26DecodePlugin] creator registered in TRT registry\n");
}

const char* Yolo26DecodePluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* Yolo26DecodePluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* Yolo26DecodePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* Yolo26DecodePluginCreator::createPlugin(
    const char* /*name*/,
    const nvinfer1::PluginFieldCollection* /*fc*/) noexcept
{
    return new Yolo26DecodePlugin();
}

nvinfer1::IPluginV2* Yolo26DecodePluginCreator::deserializePlugin(
    const char* /*name*/,
    const void* serialData,
    size_t      serialLength) noexcept
{
    std::fprintf(stderr, "[Yolo26DecodePlugin] deserializePlugin called — CUDA kernel active\n");
    return new Yolo26DecodePlugin(serialData, serialLength);
}
