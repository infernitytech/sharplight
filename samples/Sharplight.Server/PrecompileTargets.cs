using Sharplight.Hub;
using Sharplight.Packaging;
using Sharplight.Samples;

namespace Sharplight.Server;

/// <summary>The device a pulled package is precompiled for: the GPU the model will run on, with its context and batch
/// size and the server's default slot count.</summary>
internal static class PrecompileTargets
{
    /// <summary>The target, or null when nothing is precompiled (CPU runs, <c>--no-precompile</c>, no GPU).</summary>
    public static PrecompileTarget? Create(bool enabled, bool cpu, int deviceIndex, int contextLength, int batchTokens,
        IReadOnlyList<string> disabledFeatures, Action<string> log)
    {
        if (!enabled || cpu)
        {
            return null;
        }

        try
        {
            var provider = SampleRuntime.SelectProvider(cpu: false);
            var device = SampleRuntime.GetDevice(provider, deviceIndex);
            var options = new PrecompileOptions
            {
                ContextLength = contextLength,
                BatchTokens = batchTokens,
                // The batch decoder serve and run load, at the server's default slot count.
                Slots = ServeOptionSet.DefaultSlots,
                DisabledFeatures = disabledFeatures,
                // Kernels compiled now also serve later compiles; the stored sets still hold only this model's.
                UseUserKernelCache = true,
                Log = log,
            };
            return new PrecompileTarget(provider, device, options);
        }
        catch (SampleException error)
        {
            log($"Not precompiling: {error.Message}");
            return null;
        }
    }
}
