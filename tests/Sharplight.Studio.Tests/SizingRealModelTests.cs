using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The load panel's footprints from the engine on real models, read without a device (no GPU context). Gemma 4 31B Q4_K_M
/// (<c>gguf/gemma-4-31B-it-Q4_K_M.gguf</c> under the test models) with one slot takes the session path with window-sized caches: weights
/// and state are about 21 GiB at 32K and 28 GiB at 128K (benchmarks.md section 77), a q8_0 cache less; the total adds the activation bound
/// of a 512-row prompt batch (about 3 GiB).
/// </summary>
public sealed class SizingRealModelTests
{
    private const double GiB = 1L << 30;

    [Fact]
    public void Gemma4_31B_FootprintsFollowContextAndKvType()
    {
        string path = RealStudioSession.Require("gguf/gemma-4-31B-it-Q4_K_M.gguf");
        using var estimator = new FootprintEstimator(TestHost.Loader);
        var ct = TestContext.Current.CancellationToken;
        Runtime.ModelFootprint Estimate(int context, string kv) => estimator.Estimate(
            new ModelLoadRequest("gemma", path, new LoadSettings { ContextLength = context, KvCacheType = kv }, []), ct);
        static double Resident(Runtime.ModelFootprint fp) => (fp.WeightBytes + fp.StateBytes) / GiB;

        var at32 = Estimate(32768, "f16");
        var at128 = Estimate(131072, "f16");
        var at128Q8 = Estimate(131072, "q8_0");
        foreach (var (name, fp) in new[] { ("32K f16", at32), ("128K f16", at128), ("128K q8_0", at128Q8) })
        {
            TestContext.Current.TestOutputHelper?.WriteLine($"{name}: total {fp.Total / GiB:F2} GiB (weights {fp.WeightBytes / GiB:F2}, state {fp.StateBytes / GiB:F2}, activations {fp.ActivationBytes / GiB:F2})");
        }

        Assert.InRange(Resident(at32), 20, 22);
        Assert.InRange(Resident(at128), 27, 29);
        Assert.InRange(Resident(at128Q8), 22, Resident(at128) - 4);
        Assert.InRange(at32.ActivationBytes / GiB, 0.5, 4);
        Assert.Equal(0, estimator.ExpertBytes(new ModelLoadRequest("gemma", path, new LoadSettings(), []), ct));
    }

    /// <summary>
    /// On the GPU (opt-in, <c>SHARPLIGHT_STUDIO_GPU_SIZING=1</c>: it loads 18 GB of weights): what the load holds against the estimate,
    /// and the panel's verdict on this device.
    /// </summary>
    [Theory]
    [InlineData(32768, "f16")]
    [InlineData(131072, "q8_0")]
    public async Task Gemma4_31B_LoadedOnTheGpu_HoldsNoMoreThanTheEstimate(int context, string kv)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_GPU_SIZING") == "1", "SHARPLIGHT_STUDIO_GPU_SIZING=1 loads the model on the GPU");
        string path = RealStudioSession.Require("gguf/gemma-4-31B-it-Q4_K_M.gguf");
        using var session = new RealStudioSession();
        var device = session.Runtime.Devices[0];
        Assert.SkipUnless(device.IsGpu, "no GPU");
        var ct = TestContext.Current.CancellationToken;
        var request = new ModelLoadRequest("gemma", path, new LoadSettings { DeviceId = device.Id, ContextLength = context, KvCacheType = kv }, []);
        var estimate = await session.Runtime.EstimateFootprintAsync(request, ct);
        var advisor = new Services.Fit.FitAdvisor(session.Runtime, session.Services.Settings);
        var memory = await advisor.GetMemoryAsync(ct);
        var report = advisor.Plan(estimate, request.Settings, memory);
        var model = await session.Runtime.LoadAsync(request, null, ct);
        var usage = session.Runtime.GetDeviceUsage().First(entry => entry.Id == device.Id);
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"{context} {kv}: estimate {estimate.Total / GiB:F2} GiB (weights + state {(estimate.WeightBytes + estimate.StateBytes) / GiB:F2}), verdict {report.Verdict} "
            + $"against {memory.First(entry => entry.Id == device.Id).FreeBytes / GiB:F2} GiB free; loaded {model.ResidentBytes / GiB:F2} GiB, device in use {usage.UsedBytes / GiB:F2} GiB");
        Assert.True(model.ResidentBytes <= estimate.Total, $"{model.ResidentBytes} > {estimate.Total}");
        await session.Runtime.UnloadAsync("gemma");
    }

    [Fact]
    public void MixtureOfExperts_ReportsItsExpertWeights()
    {
        string path = RealStudioSession.Require("gguf/OLMoE-1B-7B-0924-Instruct/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");
        using var estimator = new FootprintEstimator(TestHost.Loader);
        var request = new ModelLoadRequest("olmoe", path, new LoadSettings { ContextLength = 4096 }, []);
        var ct = TestContext.Current.CancellationToken;
        long experts = estimator.ExpertBytes(request, ct);
        long weights = estimator.Estimate(request, ct).WeightBytes;
        TestContext.Current.TestOutputHelper?.WriteLine($"experts {experts / GiB:F2} GiB of {weights / GiB:F2} GiB of weights");
        // Nearly all of OLMoE's weights are its 64 experts per layer.
        Assert.InRange(experts, weights * 0.7, weights);
    }
}
