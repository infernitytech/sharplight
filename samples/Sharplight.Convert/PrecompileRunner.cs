using Sharplight.Import.Onnx;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>
/// Runs <c>precompile</c>: the two precompilation stages of the execution providers (portable code, then device code) through
/// <see cref="PackagePrecompiler"/>, for one device, every device, or a device profile, and appends the artifact sets.
/// </summary>
internal sealed class PrecompileRunner(TextWriter output)
{
    private const long DefaultOnnxBatch = 1;

    public async Task RunAsync(PrecompileSettings settings, CancellationToken cancellationToken)
    {
        // ONNX graphs run with the CPU as fallback for data-dependent ops, so they are compiled that way too.
        using var cpu = settings.Profile is null && settings.Device is not { Provider: "cpu" } ? DeviceOpener.OpenCpu() : null;
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

        PackagePrecompileReport report;
        try
        {
            report = await PrecompileAsync(settings, options, cancellationToken);
        }
        catch (NotSupportedException error)
        {
            throw new SampleException(error.Message, ExitCodes.InvalidInput);
        }
        catch (MissingSymbolBoundException error)
        {
            throw new SampleException($"{error.Message} Give it with --bound SYMBOL=VALUE.", ExitCodes.InvalidInput);
        }

        Print(report);
    }

    private static Task<PackagePrecompileReport> PrecompileAsync(PrecompileSettings settings, PrecompileOptions options, CancellationToken cancellationToken)
    {
        if (settings.Profile is { } profile)
        {
            return PackagePrecompiler.PrecompileAsync(settings.PackagePath, DeviceOpener.ForProfile(profile), profile, options, null, cancellationToken);
        }

        if (settings.Device is { } spec)
        {
            var (provider, device) = DeviceOpener.Find(spec);
            return PackagePrecompiler.PrecompileAsync(settings.PackagePath, provider, device, options, null, cancellationToken);
        }

        return PackagePrecompiler.PrecompileAsync(settings.PackagePath, DeviceOpener.All(), options, null, cancellationToken);
    }

    private void Print(PackagePrecompileReport report)
    {
        foreach (var device in report.Devices)
        {
            output.WriteLine($"{device.Provider} {device.Device}:");
            output.WriteLine($"  portable: {Describe(device.Portable)}");
            output.WriteLine($"  device:   {Describe(device.DeviceCode)}");
            if (device.Note is { } note)
            {
                output.WriteLine($"  {note}");
            }
        }

        foreach (var skipped in report.Skipped)
        {
            output.WriteLine($"Skipped {skipped.Provider}{(skipped.Device.Length == 0 ? "" : " " + skipped.Device)}: {skipped.Reason}");
        }

        output.WriteLine($"slang {Shaders.SlangCompiler.Statistics}");
        output.WriteLine(report.StoredSets.Count == 0
            ? "Nothing new to store."
            : $"{report.PackagePath}: {ByteSize.Format(report.PackageBytes)} ({ByteSize.Format(report.StoredBytes)} of artifacts added) in {report.Elapsed.TotalSeconds:F1} s");
    }

    private static string Describe(CodeSetSummary? set)
    {
        if (set is null)
        {
            return "none";
        }

        var shared = set.SharedWith is null ? $"{set.Elapsed.TotalMilliseconds:F0} ms" : $"shared with {set.SharedWith}";
        return $"{set.Kind}, {set.Entries} entries, {ByteSize.Format(set.Bytes)} ({set.ReusedEntries} already stored; {shared})";
    }

    /// <summary>An ONNX model's batch dimension is bounded to 1 unless <c>--bound</c> gives the batch it will run with.</summary>
    private static IReadOnlyDictionary<string, long> OnnxBatchBound(ModelDefinition definition)
    {
        return definition.Properties.TryGetValue(OnnxGraphAdapter.BatchSymbolProperty, out var batch) && batch is string symbol
            ? new Dictionary<string, long> { [symbol] = DefaultOnnxBatch }
            : new Dictionary<string, long>();
    }
}
