#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <string>

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

class Yolo26DecodePlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    Yolo26DecodePlugin() = default;
    explicit Yolo26DecodePlugin(const void* data, size_t length);

    // IPluginV2 identity
    const char* getPluginType()    const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int         getNbOutputs()     const noexcept override { return 1; }
    int         initialize()             noexcept override { return 0; }
    void        terminate()              noexcept override {}
    void        destroy()                noexcept override { delete this; }
    size_t      getSerializationSize()   const noexcept override { return 0; }
    void        serialize(void*)         const noexcept override {}
    void        setPluginNamespace(const char* ns) noexcept override { mNamespace = ns; }
    const char* getPluginNamespace()     const noexcept override { return mNamespace.c_str(); }

    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType(
        int index,
        const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept override;

    // IPluginV2DynamicExt
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs,
        int nbOutputs) noexcept override;

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,  int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,  int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override
    { return 0; }

    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const*       outputs,
        void*              workspace,
        cudaStream_t       stream) noexcept override;

private:
    std::string mNamespace;
};

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------

class Yolo26DecodePluginCreator final : public nvinfer1::IPluginCreator
{
public:
    Yolo26DecodePluginCreator();

    const char* getPluginName()      const noexcept override;
    const char* getPluginVersion()   const noexcept override;
    const char* getPluginNamespace() const noexcept override { return ""; }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        const char*                            name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t      serialLength) noexcept override;

    void setPluginNamespace(const char* ns) noexcept override {}

private:
    nvinfer1::PluginFieldCollection mFC{};
};

REGISTER_TENSORRT_PLUGIN(Yolo26DecodePluginCreator);
