using System.Runtime.InteropServices;
using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>
/// Creates the compute pipeline of a kernel the way every path needs it (from SPIR-V through a pipeline cache, from stored pipeline
/// binaries, or capturing pipeline binaries for storage), with one create info, so the pipeline key a stored binary was captured
/// under is the key a later load computes. Thread-safe.
/// </summary>
internal sealed unsafe class PipelineBuilder
{
    /// <summary>Slang names the single SPIR-V entry point "main" regardless of the source function name. Lives as long as the process.</summary>
    private static readonly byte* s_entryPoint = (byte*)Marshal.StringToHGlobalAnsi("main");

    private readonly VkDeviceApi _api;
    private readonly VkPipelineLayout _layout;
    private readonly uint _subgroupSize;
    private readonly bool _requireSubgroupSize;
    private readonly bool _fullSubgroups;
    private readonly VkPipelineCreateFlags _flags;
    private readonly bool _portabilityRewrite;

    /// <param name="api">The device functions.</param>
    /// <param name="layout">The pipeline layout every kernel shares.</param>
    /// <param name="subgroupSize">Subgroup size kernels are specialized for.</param>
    /// <param name="requireSubgroupSize">Pin the subgroup size (subgroup size control enabled and the device has a range).</param>
    /// <param name="fullSubgroups">Require full subgroups (subgroup size control enabled).</param>
    /// <param name="flags">Extra create flags (diagnostics: statistics capture).</param>
    /// <param name="portabilityRewrite">Apply <see cref="SpirvPortability.Rewrite"/> to every module (layered implementations such as MoltenVK).</param>
    public PipelineBuilder(VkDeviceApi api, VkPipelineLayout layout, uint subgroupSize, bool requireSubgroupSize, bool fullSubgroups, VkPipelineCreateFlags flags,
        bool portabilityRewrite)
    {
        _api = api;
        _layout = layout;
        _subgroupSize = subgroupSize;
        _requireSubgroupSize = requireSubgroupSize;
        _fullSubgroups = fullSubgroups;
        _flags = flags;
        _portabilityRewrite = portabilityRewrite;
    }

    /// <summary>Creates the shader module of <paramref name="spirv"/>.</summary>
    public VkShaderModule CreateModule(byte[] spirv)
    {
        if (_portabilityRewrite)
        {
            // Layered implementations (MoltenVK) translate SPIR-V to another language; avoid constructs they mistranslate.
            spirv = SpirvPortability.Rewrite(spirv);
        }

        VkShaderModule module;
        fixed (byte* code = spirv)
        {
            var moduleInfo = new VkShaderModuleCreateInfo { codeSize = (nuint)spirv.Length, pCode = (uint*)code };
            _api.vkCreateShaderModule(&moduleInfo, null, &module).CheckResult();
        }

        return module;
    }

    /// <summary>Creates the pipeline of <paramref name="module"/>, recorded in <paramref name="pipelineCache"/> (may be null).</summary>
    /// <exception cref="VkException">The driver failed to create it.</exception>
    public VkPipeline Create(VkShaderModule module, VkPipelineCache pipelineCache)
    {
        var subgroup = new VkPipelineShaderStageRequiredSubgroupSizeCreateInfo { requiredSubgroupSize = _subgroupSize };
        var info = CreateInfo(module, &subgroup);
        VkPipeline pipeline;
        _api.vkCreateComputePipelines(pipelineCache, 1, &info, null, &pipeline).CheckResult();
        return pipeline;
    }

    /// <summary>The global pipeline binary key of the driver (VK_KHR_pipeline_binary must be enabled), hexadecimal.</summary>
    public string GlobalBinaryKey()
    {
        var key = new VkPipelineBinaryKeyKHR();
        _api.vkGetPipelineKeyKHR(null, &key).CheckResult();
        return Convert.ToHexString(new ReadOnlySpan<byte>(key.key, (int)key.keySize));
    }

    /// <summary>
    /// Creates the pipeline of <paramref name="module"/> from stored binaries (VK_KHR_pipeline_binary must be enabled), without
    /// compiling. Only when <paramref name="entry"/> was captured under the key of this create info and the driver accepts the
    /// binaries; otherwise nothing is created.
    /// </summary>
    /// <returns>The pipeline, or <see cref="VkPipeline.Null"/> when the binaries do not fit (the caller compiles instead).</returns>
    public VkPipeline TryCreateFromBinaries(VkShaderModule module, PipelineBinaryEntry entry)
    {
        var subgroup = new VkPipelineShaderStageRequiredSubgroupSizeCreateInfo { requiredSubgroupSize = _subgroupSize };
        var info = CreateInfo(module, &subgroup);
        if (!PipelineKey(&info).AsSpan().SequenceEqual(entry.PipelineKey))
        {
            return VkPipeline.Null;
        }

        int count = entry.Binaries.Count;
        var keys = new VkPipelineBinaryKeyKHR[count];
        var data = new VkPipelineBinaryDataKHR[count];
        var handles = new VkPipelineBinaryKHR[count];
        var pins = new GCHandle[count];
        try
        {
            for (int i = 0; i < count; i++)
            {
                var (key, bytes) = entry.Binaries[i];
                keys[i] = new VkPipelineBinaryKeyKHR { keySize = (uint)key.Length };
                pins[i] = GCHandle.Alloc(bytes, GCHandleType.Pinned);
                data[i] = new VkPipelineBinaryDataKHR { dataSize = (nuint)bytes.Length, pData = (void*)pins[i].AddrOfPinnedObject() };
            }

            fixed (VkPipelineBinaryKeyKHR* keyPointer = keys)
            fixed (VkPipelineBinaryDataKHR* dataPointer = data)
            fixed (VkPipelineBinaryKHR* handlePointer = handles)
            {
                for (int i = 0; i < count; i++)
                {
                    entry.Binaries[i].Key.CopyTo(new Span<byte>(keyPointer[i].key, PipelineBinaryCodec.MaxKeySize));
                }

                var keysAndData = new VkPipelineBinaryKeysAndDataKHR { binaryCount = (uint)count, pPipelineBinaryKeys = keyPointer, pPipelineBinaryData = dataPointer };
                var createInfo = new VkPipelineBinaryCreateInfoKHR { pKeysAndDataInfo = &keysAndData };
                var handlesInfo = new VkPipelineBinaryHandlesInfoKHR { pipelineBinaryCount = (uint)count, pPipelineBinaries = handlePointer };
                if (_api.vkCreatePipelineBinariesKHR(&createInfo, null, &handlesInfo) != VkResult.Success || handlesInfo.pipelineBinaryCount != count)
                {
                    return VkPipeline.Null;
                }

                var binaryInfo = new VkPipelineBinaryInfoKHR { binaryCount = (uint)count, pPipelineBinaries = handlePointer };
                info.pNext = &binaryInfo;
                VkPipeline pipeline;
                // Pipelines created from binaries take no pipeline cache.
                var result = _api.vkCreateComputePipelines(VkPipelineCache.Null, 1, &info, null, &pipeline);
                return result == VkResult.Success ? pipeline : VkPipeline.Null;
            }
        }
        finally
        {
            foreach (var handle in handles.Where(h => h != VkPipelineBinaryKHR.Null))
            {
                _api.vkDestroyPipelineBinaryKHR(handle, null);
            }

            foreach (var pin in pins.Where(p => p.IsAllocated))
            {
                pin.Free();
            }
        }
    }

    /// <summary>
    /// Compiles the pipeline of <paramref name="module"/> and captures the driver binaries it consists of, with the key of the create
    /// info a load uses (VK_KHR_pipeline_binary must be enabled). The pipeline is destroyed again.
    /// </summary>
    /// <exception cref="VkException">The driver failed to create the pipeline or its binaries.</exception>
    public PipelineBinaryEntry Capture(VkShaderModule module)
    {
        var subgroup = new VkPipelineShaderStageRequiredSubgroupSizeCreateInfo { requiredSubgroupSize = _subgroupSize };
        var info = CreateInfo(module, &subgroup);
        var pipelineKey = PipelineKey(&info);
        var flags = new VkPipelineCreateFlags2CreateInfo { flags = VkPipelineCreateFlags2.CaptureDataKHR };
        info.pNext = &flags;
        VkPipeline pipeline;
        _api.vkCreateComputePipelines(VkPipelineCache.Null, 1, &info, null, &pipeline).CheckResult();
        VkPipelineBinaryKHR[] handles = [];
        try
        {
            var createInfo = new VkPipelineBinaryCreateInfoKHR { pipeline = pipeline };
            var query = new VkPipelineBinaryHandlesInfoKHR();
            _api.vkCreatePipelineBinariesKHR(&createInfo, null, &query).CheckResult();
            handles = new VkPipelineBinaryKHR[query.pipelineBinaryCount];
            fixed (VkPipelineBinaryKHR* handlePointer = handles)
            {
                var handlesInfo = new VkPipelineBinaryHandlesInfoKHR { pipelineBinaryCount = (uint)handles.Length, pPipelineBinaries = handlePointer };
                _api.vkCreatePipelineBinariesKHR(&createInfo, null, &handlesInfo).CheckResult();
            }

            return new PipelineBinaryEntry(pipelineKey, [.. handles.Select(ReadBinary)]);
        }
        finally
        {
            var release = new VkReleaseCapturedPipelineDataInfoKHR { pipeline = pipeline };
            _api.vkReleaseCapturedPipelineDataKHR(&release, null);
            foreach (var handle in handles.Where(h => h != VkPipelineBinaryKHR.Null))
            {
                _api.vkDestroyPipelineBinaryKHR(handle, null);
            }

            _api.vkDestroyPipeline(pipeline, null);
        }
    }

    private (byte[] Key, byte[] Data) ReadBinary(VkPipelineBinaryKHR binary)
    {
        var info = new VkPipelineBinaryDataInfoKHR { pipelineBinary = binary };
        var key = new VkPipelineBinaryKeyKHR();
        nuint size = 0;
        _api.vkGetPipelineBinaryDataKHR(&info, &key, &size, null).CheckResult();
        var data = new byte[size];
        fixed (byte* pointer = data)
        {
            _api.vkGetPipelineBinaryDataKHR(&info, &key, &size, pointer).CheckResult();
        }

        return (new ReadOnlySpan<byte>(key.key, (int)key.keySize).ToArray(), data.AsSpan(0, (int)size).ToArray());
    }

    private byte[] PipelineKey(VkComputePipelineCreateInfo* info)
    {
        var createInfo = new VkPipelineCreateInfoKHR { pNext = info };
        var key = new VkPipelineBinaryKeyKHR();
        _api.vkGetPipelineKeyKHR(&createInfo, &key).CheckResult();
        return new ReadOnlySpan<byte>(key.key, (int)key.keySize).ToArray();
    }

    /// <summary>The create info of every pipeline: one shared layout, the entry point, the subgroup size settings.</summary>
    private VkComputePipelineCreateInfo CreateInfo(VkShaderModule module, VkPipelineShaderStageRequiredSubgroupSizeCreateInfo* subgroup)
    {
        var stage = new VkPipelineShaderStageCreateInfo
        {
            pNext = _requireSubgroupSize ? subgroup : null,
            stage = VkShaderStageFlags.Compute,
            module = module,
            pName = s_entryPoint,
            flags = _fullSubgroups ? VkPipelineShaderStageCreateFlags.RequireFullSubgroups : 0,
        };
        return new VkComputePipelineCreateInfo { layout = _layout, stage = stage, flags = _flags };
    }
}
