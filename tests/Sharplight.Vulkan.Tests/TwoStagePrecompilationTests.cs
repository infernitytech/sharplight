using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The two precompilation stages of the Vulkan provider: SPIR-V for a device profile without the driver, then device code from the
/// driver (kernel binaries, or a pipeline cache without them), and loads that use the best stored stage and ignore stored device code
/// that is damaged, foreign or stale.
/// </summary>
public sealed class TwoStagePrecompilationTests
{
    private const int Hidden = 128;
    private const int Ffn = 256;
    private const int Tokens = 4;

    private static readonly Lazy<ComputeGraph> s_graph = new(BuildGraph);

    [Fact]
    public async Task PortableStageGeneratesSpirvForTheProfileWithoutTheDriver()
    {
        Gpu.Require();
        var provider = new VulkanProvider();
        var portable = await provider.CompilePortableAsync(new PortableCompilationRequest(Gpu.Vulkan!.CreateProfile(), Workload), TestContext.Current.CancellationToken);

        Assert.Equal("spirv_1_6", provider.PortableCodeFormat);
        Assert.Equal(provider.PortableCodeFormat, portable.Format);
        Assert.Equal("vulkan", portable.Provider);
        Assert.Equal(ArtifactTiers.Code, ArtifactDescriptor.Get(portable.Descriptor, ArtifactDescriptorKeys.Tier));
        Assert.NotEmpty(portable.Entries);
        Assert.All(portable.Entries.Keys, key => Assert.StartsWith("kernel/", key));
        // SPIR-V magic number.
        Assert.All(portable.Entries.Values, code => Assert.Equal(0x07230203u, MemoryMarshal.Read<uint>(code.Span)));
    }

    [Fact]
    public async Task DeviceStageProducesKernelBinariesThatLoadsUseWithoutCompiling()
    {
        Gpu.Require();
        var (provider, info) = Device();
        var portable = await PortableAsync(provider);
        using var device = provider.OpenDevice(info);
        var deviceCode = await provider.CompileDeviceSpecificAsync(device, portable, TestContext.Current.CancellationToken);
        Assert.False(deviceCode.IsEmpty);
        Assert.Equal(ArtifactTiers.Binary, ArtifactDescriptor.Get(deviceCode.Descriptor, ArtifactDescriptorKeys.Tier));
        Assert.Equal(info.Name, deviceCode.Device);
        var vulkan = (VulkanDevice)device;
        Assert.SkipUnless(vulkan.ArtifactTargets.Any(t => t.Kind == VulkanDevice.PipelineBinaryKind), "The driver has no VK_KHR_pipeline_binary.");
        Assert.Equal(VulkanDevice.PipelineBinaryKind, deviceCode.Kind);
        Assert.Equal(portable.Entries.Count, deviceCode.Entries.Count);
        Assert.All(deviceCode.Entries.Keys, key => Assert.StartsWith(VulkanDevice.PipelineBinaryPrefix, key));

        var expected = Run(provider.OpenDevice(info), new CompilationArtifacts(), out _);
        var actual = Run(provider.OpenDevice(info), new CompilationArtifacts(Store(portable, deviceCode)), out var loaded);
        Assert.Equal(portable.Entries.Count, loaded.PipelinesFromBinaries);
        Assert.Equal(0, loaded.RejectedDeviceCode);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public async Task DamagedForeignOrStaleKernelBinariesAreIgnored()
    {
        Gpu.Require();
        var (provider, info) = Device();
        var portable = await PortableAsync(provider);
        DeviceCodeSet deviceCode;
        using (var device = provider.OpenDevice(info))
        {
            Assert.SkipUnless(device.ArtifactTargets.Any(t => t.Kind == VulkanDevice.PipelineBinaryKind), "The driver has no VK_KHR_pipeline_binary.");
            deviceCode = await provider.CompileDeviceSpecificAsync(device, portable, TestContext.Current.CancellationToken);
        }

        var expected = Run(provider.OpenDevice(info), new CompilationArtifacts(), out _);

        // A flipped byte in the binary data fails the checksum: never handed to the driver.
        var damaged = WithEntries(deviceCode, data =>
        {
            var copy = data.ToArray();
            copy[^1] ^= 0x5A;
            return copy;
        });
        Assert.Equal(expected, Run(provider.OpenDevice(info), new CompilationArtifacts(Store(portable, damaged)), out var afterDamage));
        Assert.Equal(0, afterDamage.PipelinesFromBinaries);
        Assert.Equal(portable.Entries.Count, afterDamage.RejectedDeviceCode);

        // Intact binaries captured under another pipeline key (another create info) are refused before the driver sees them.
        var foreign = WithEntries(deviceCode, data =>
        {
            Assert.True(PipelineBinaryCodec.TryDecode(data.Span, out var entry));
            var key = entry!.PipelineKey.ToArray();
            key[0] ^= 0xFF;
            return PipelineBinaryCodec.Encode(entry with { PipelineKey = key });
        });
        Assert.Equal(expected, Run(provider.OpenDevice(info), new CompilationArtifacts(Store(portable, foreign)), out var afterForeign));
        Assert.Equal(0, afterForeign.PipelinesFromBinaries);
        Assert.Equal(portable.Entries.Count, afterForeign.RejectedDeviceCode);

        // A set for another driver build (binary key) does not match the device at all.
        var descriptor = new Dictionary<string, string>(deviceCode.Descriptor) { [ArtifactDescriptorKeys.BinaryKey] = "00" };
        var stale = new DeviceCodeSet(deviceCode.Provider, deviceCode.Device, descriptor, deviceCode.Entries);
        Assert.Equal(expected, Run(provider.OpenDevice(info), new CompilationArtifacts(Store(portable, stale)), out var afterStale));
        Assert.Equal(0, afterStale.PipelinesFromBinaries);
        Assert.Equal(0, afterStale.RejectedDeviceCode);
    }

    [Fact]
    public async Task WithoutKernelBinariesTheDeviceCodeIsAValidatedPipelineCache()
    {
        Gpu.Require();
        var (provider, info) = Device();
        var portable = await PortableAsync(provider);
        var options = new DeviceOptions();
        options.DisabledFeatures.Add(DeviceFeatures.KernelBinaries);
        using var device = (VulkanDevice)provider.OpenDevice(info, options);
        Assert.DoesNotContain(device.ArtifactTargets, t => t.Kind == VulkanDevice.PipelineBinaryKind);

        var deviceCode = await provider.CompileDeviceSpecificAsync(device, portable, TestContext.Current.CancellationToken);
        Assert.Equal(VulkanDevice.PipelineCacheKind, deviceCode.Kind);
        var blob = Assert.Single(deviceCode.Entries);
        Assert.Equal(VulkanDevice.PipelineCacheKey, blob.Key);
        Assert.True(PipelineCacheHeader.Matches(blob.Value.Span, device.Phys.VendorId, device.Phys.DeviceId, device.Phys.PipelineCacheUuid));
        Assert.False(PipelineCacheHeader.Matches(blob.Value.Span, device.Phys.VendorId, device.Phys.DeviceId + 1, device.Phys.PipelineCacheUuid));

        var expected = Run(provider.OpenDevice(info, options), new CompilationArtifacts(), out _);
        Assert.Equal(expected, Run(provider.OpenDevice(info, options), new CompilationArtifacts(Store(portable, deviceCode)), out var loaded));
        Assert.Equal(0, loaded.RejectedDeviceCode);
    }

    [Fact]
    public async Task DeviceStageRejectsPortableCodeOfAnotherProfile()
    {
        Gpu.Require();
        var (provider, info) = Device();
        var own = await PortableAsync(provider);
        // The same code labelled for a device with another subgroup size.
        var descriptor = new Dictionary<string, string>(own.Descriptor) { [ArtifactDescriptorKeys.Subgroup] = "7" };
        var portable = new PortableCodeSet(own.Provider, descriptor, own.Entries);
        using var device = provider.OpenDevice(info);
        await Assert.ThrowsAsync<ArgumentException>(() => provider.CompileDeviceSpecificAsync(device, portable, TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<ArgumentException>(() => provider.CompileDeviceSpecificAsync(Gpu.Cpu, portable, TestContext.Current.CancellationToken));
    }

    private static (VulkanProvider Provider, DeviceInfo Info) Device()
    {
        var provider = new VulkanProvider();
        return (provider, provider.GetDevices().First(d => d.Name == Gpu.Vulkan!.Info.Name));
    }

    private static Task<PortableCodeSet> PortableAsync(VulkanProvider provider)
        => provider.CompilePortableAsync(new PortableCompilationRequest(Gpu.Vulkan!.CreateProfile(), Workload), TestContext.Current.CancellationToken);

    private static void Workload(IComputeDevice device, CompilationArtifacts artifacts, CancellationToken cancellationToken)
    {
        using (Compile(device, artifacts))
        {
        }
    }

    /// <summary>Compiles and runs the graph on <paramref name="device"/> (disposed afterwards) and returns its output.</summary>
    private static float[] Run(IComputeDevice device, CompilationArtifacts artifacts, out (int PipelinesFromBinaries, int RejectedDeviceCode) counters)
    {
        using (device)
        {
            using var model = Compile(device, artifacts);
            var input = new float[Tokens * Hidden];
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = MathF.Sin(i * 0.37f);
            }

            model.SetInput(s_graph.Value.Inputs.Single(), MemoryMarshal.AsBytes(input.AsSpan()));
            var parameters = new RunParameters();
            parameters.Symbols["T"] = Tokens;
            model.Run(parameters);
            var output = new float[Tokens * Hidden];
            model.GetOutput<float>("h", output);
            var vulkan = (VulkanDevice)device;
            counters = (vulkan.PipelinesFromBinaries, vulkan.RejectedDeviceCode);
            return output;
        }
    }

    private static CompiledModel Compile(IComputeDevice device, CompilationArtifacts artifacts) => CompiledModel.Compile(s_graph.Value, new PlacementOptions
    {
        Devices = [new DeviceSlot { Device = device }],
        Compile = new CompileOptions { MaxTokens = 16, ContextLength = 64, CacheFor = artifacts.For },
        ReservePerDevice = 0,
    });

    private static MemoryStore Store(params CodeSet[] sets) => new([.. sets.Select(set => set.ToArtifactSet())]);

    private static DeviceCodeSet WithEntries(DeviceCodeSet set, Func<ReadOnlyMemory<byte>, byte[]> change)
        => new(set.Provider, set.Device, set.Descriptor, set.Entries.ToDictionary(e => e.Key, e => (ReadOnlyMemory<byte>)change(e.Value)));

    /// <summary>A gated MLP: an f16 and a q8_0 matmul.</summary>
    private static ComputeGraph BuildGraph()
    {
        var tg = new TG("two-stage");
        var x = tg.InF("x", TG.T, Hidden);
        var up = tg.RandW("up", DataType.F16, 0.3f, Ffn, Hidden);
        var down = tg.RandW("down", DataType.Q8_0, 0.3f, Hidden, Ffn);
        tg.B.Output("h", tg.B.Add(tg.B.MatMul(tg.B.Silu(tg.B.MatMul(x, up)), down), x));
        return tg.B.Build();
    }
}
