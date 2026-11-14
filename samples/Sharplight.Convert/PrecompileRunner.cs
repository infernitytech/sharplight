using Sharplight.Models;
using Sharplight.Onnx;
using Sharplight.Packaging;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>
/// Runs <c>precompile</c>: compiles the package on a live device or for a device profile through <see cref="PackagePrecompiler"/>
/// and appends the artifact sets.
/// </summary>
internal sealed class PrecompileRunner(TextWriter output)
{
    private const long DefaultOnnxBatch = 1;

    public void Run(PrecompileSettings settings)
    {
        // ONNX graphs run with the CPU as fallback for data-dependent ops, so they are compiled that way too.
        using var cpu = settings.Profile is null && settings.Device is { Provider: not "cpu" } ? DeviceOpener.OpenCpu() : null;
        var options = new PrecompileOptions
        {
            ContextLength = settings.ContextLength,
            BatchTokens = settings.BatchTokens,
            Slots = settings.Slots,
            KvCacheTokens = settings.KvCacheTokens,
            DraftTokens = settings.DraftTokens,
            SymbolBounds = settings.SymbolBounds,
            DefaultSymbolBounds = OnnxBatchBound,
            FallbackDevice = cpu,
            Tiers = settings.Tiers,
            DisabledFeatures = settings.DisabledFeatures,
            UseUserKernelCache = settings.UseUserKernelCache,
            Log = output.WriteLine,
        };

        PrecompileResult result;
        try
        {
            result = settings.Profile is { } profile
                ? PackagePrecompiler.Precompile(settings.PackagePath, profile, options)
                : PackagePrecompiler.Precompile(settings.PackagePath, deviceOptions => DeviceOpener.Open(settings.Device!.Value, deviceOptions), options);
        }
        catch (NotSupportedException error)
        {
            throw new SampleException(error.Message, ExitCodes.InvalidInput);
        }
        catch (MissingSymbolBoundException error)
        {
            throw new SampleException($"{error.Message} Give it with --bound SYMBOL=VALUE.", ExitCodes.InvalidInput);
        }

        output.WriteLine($"slang {Shaders.SlangCompiler.Statistics}");
        output.WriteLine(result.StoredSets.Count == 0
            ? "Nothing new to store."
            : $"{settings.PackagePath}: {ByteSize.Format(result.PackageBytes)} ({ByteSize.Format(result.StoredBytes)} of artifacts added)");
    }

    /// <summary>An ONNX model's batch dimension is bounded to 1 unless <c>--bound</c> gives the batch it will run with.</summary>
    private static IReadOnlyDictionary<string, long> OnnxBatchBound(ModelDefinition definition)
    {
        return definition.Properties.TryGetValue(OnnxGraphAdapter.BatchSymbolProperty, out var batch) && batch is string symbol
            ? new Dictionary<string, long> { [symbol] = DefaultOnnxBatch }
            : new Dictionary<string, long>();
    }
}
