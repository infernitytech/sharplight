using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Offline compilation for a <see cref="DeviceProfile"/>: a profile exported from the live device and opened without the driver
/// lowers to exactly the kernels (cache keys) and code-tier descriptor the live device produces, with and without disabled features.
/// </summary>
public sealed class DeviceProfileTests
{
    /// <summary>Matrix units off: the portable kernels of a profile without them.</summary>
    private static readonly string[] s_matrixFeatures = [DeviceFeatures.MatrixF16, DeviceFeatures.MatrixBF16, DeviceFeatures.MatrixInt8, DeviceFeatures.MatrixWorkgroup];

    [Fact]
    public void ProfileRoundTripsThroughJson()
    {
        Gpu.Require();
        var profile = Gpu.Vulkan!.CreateProfile()!;
        var copy = DeviceProfile.FromJson(profile.ToJson());
        Assert.Equal(profile.ToJson(), copy.ToJson());
        Assert.Equal(profile.Features, copy.Features);
        Assert.Equal(profile.Kind, copy.Kind);
    }

    [Fact]
    public void ProfileDeviceProducesTheLiveDevicesKernels() => AssertSameKernels([]);

    [Fact]
    public void ReducedProfileProducesTheKernelsOfTheDeviceWithFeaturesDisabled() => AssertSameKernels(s_matrixFeatures);

    [Fact]
    public void ProfileDeviceCannotRunAndProducesOnlyTheCodeTier()
    {
        Gpu.Require();
        var provider = new VulkanProvider();
        using var offline = provider.OpenProfile(Gpu.Vulkan!.CreateProfile()!);
        Assert.Equal([ArtifactTiers.Code], offline.ArtifactTargets.Select(t => t.Tier));
        using var compiled = Compile(BuildGraph(), offline, new CompilationArtifacts());
        Assert.Throws<NotSupportedException>(() => compiled.Run(Parameters()));
    }

    private static void AssertSameKernels(string[] disabled)
    {
        Gpu.Require();
        var provider = new VulkanProvider();
        var info = provider.GetDevices().First(d => d.Name == Gpu.Vulkan!.Info.Name);
        var options = new DeviceOptions();
        foreach (var feature in disabled)
        {
            options.DisabledFeatures.Add(feature);
        }

        using var live = provider.OpenDevice(info, options);
        // The reduced profile comes from the full one, as a shipped "no matrix units" profile would.
        var profile = DeviceProfile.FromJson(Gpu.Vulkan!.CreateProfile()!.WithoutFeatures(disabled).ToJson());
        using var offline = provider.OpenProfile(profile);

        var graph = BuildGraph();
        var liveArtifacts = new CompilationArtifacts();
        var offlineArtifacts = new CompilationArtifacts();
        using (Compile(graph, live, liveArtifacts))
        {
        }

        using (Compile(graph, offline, offlineArtifacts))
        {
        }

        var liveCode = CodeTier(liveArtifacts);
        var offlineCode = CodeTier(offlineArtifacts);
        Assert.Equal(live.ArtifactTargets.Single(t => t.Tier == ArtifactTiers.Code).Id, offline.ArtifactTargets.Single().Id);
        Assert.Equal(ArtifactDescriptor.Format(liveCode.Descriptor), ArtifactDescriptor.Format(offlineCode.Descriptor));
        Assert.NotEmpty(liveCode.Entries);
        Assert.Equal(liveCode.Entries.Keys.Order(StringComparer.Ordinal), offlineCode.Entries.Keys.Order(StringComparer.Ordinal));
        foreach (var (key, code) in liveCode.Entries)
        {
            Assert.True(code.Span.SequenceEqual(offlineCode.Entries[key].Span), $"{key} differs");
        }
    }

    private static CompilationCache CodeTier(CompilationArtifacts artifacts)
        => artifacts.Recorded.Single(s => ArtifactDescriptor.Get(s.Descriptor, ArtifactDescriptorKeys.Tier) == ArtifactTiers.Code);

    private static CompiledModel Compile(ComputeGraph graph, IComputeDevice device, CompilationArtifacts artifacts) => CompiledModel.Compile(graph, new PlacementOptions
    {
        Devices = [new DeviceSlot { Device = device }],
        Compile = new CompileOptions { MaxTokens = 64, ContextLength = 256, SymbolBounds = new Dictionary<string, long> { ["R"] = 64 }, CacheFor = artifacts.For },
        ReservePerDevice = 0,
    });

    private static RunParameters Parameters()
    {
        var parameters = new RunParameters();
        parameters.Symbols["T"] = 1;
        parameters.Symbols["R"] = 1;
        parameters.Scalars["n_past"] = 0;
        return parameters;
    }

    /// <summary>F16 matmuls large enough for matrix-unit kernels, plus cached attention.</summary>
    private static ComputeGraph BuildGraph()
    {
        const int H = 256, F = 512, Hq = 4, Hkv = 2, D = 64;
        var tg = new TG("profile");
        var x = tg.InF("x", TG.T, H);
        var w1 = tg.RandW("w1", DataType.F16, 0.3f, F, H);
        var w2 = tg.RandW("w2", DataType.Q8_0, 0.3f, H, F);
        var w3 = tg.RandW("w3", DataType.F16, 0.3f, Hkv * D, H);
        var q = tg.InF("q", TG.T, Hq, D);
        var kc = tg.State("kc", DataType.F16, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", DataType.F16, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        var h = tg.B.Add(tg.B.MatMul(tg.B.Silu(tg.B.MatMul(x, w1)), w2), x);
        var kv = tg.B.Reshape(tg.B.MatMul(h, w3), 0, Hkv, D);
        tg.B.CacheWrite(kc, kv);
        tg.B.CacheWrite(vc, kv);
        tg.B.Output("h", h);
        tg.B.Output("o", tg.B.CachedAttention(q, kc, vc, 0.125f));
        return tg.B.Build();
    }
}
