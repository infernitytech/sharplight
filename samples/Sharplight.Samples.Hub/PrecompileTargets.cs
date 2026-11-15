using Sharplight.Hub;
using Sharplight.Package;

namespace Sharplight.Samples.Hub;

/// <summary>
/// What a pulled package is precompiled for: every device of the GPU provider (portable code, then device code; identical GPUs share
/// it), with the context and batch size the model will run with, and the serving batch decoder at the server's default slot count.
/// </summary>
public static class PrecompileTargets
{
    /// <summary>The <c>sharplight-server --slots</c> default; pulled packages hold the kernels of a batch decoder with this many slots.</summary>
    public const int ServeSlots = 4;

    /// <summary>The target, or null when nothing is precompiled (CPU runs, <c>--no-precompile</c>, no GPU).</summary>
    public static PrecompileTarget? Create(bool enabled, bool cpu, int contextLength, int batchTokens,
        IReadOnlyList<string> disabledFeatures, Action<string> log)
    {
        if (!enabled || cpu)
        {
            return null;
        }

        try
        {
            var provider = SampleRuntime.SelectProvider(cpu: false);
            var options = new PrecompileOptions
            {
                ContextLength = contextLength,
                BatchTokens = batchTokens,
                Slots = ServeSlots,
                DisabledFeatures = disabledFeatures,
                // Kernels compiled now also serve later compiles; the stored sets still hold only this model's.
                UseUserKernelCache = true,
                Log = log,
            };
            return new PrecompileTarget([provider], options);
        }
        catch (SampleException error)
        {
            log($"Not precompiling: {error.Message}");
            return null;
        }
    }
}
