using System.Text.RegularExpressions;
using Sharplight.Hub;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Hub;

/// <summary>
/// The Discover filters the Hub cannot apply in a search: gating, quantization, parameter range and "fits this machine". They run on
/// the search results (<see cref="HubModelSummary"/>); the task, format, license and sort order go to the Hub in the query
/// (<see cref="ToQuery"/>).
/// </summary>
internal static partial class DiscoverFilter
{
    /// <summary>Bytes per parameter of the smallest download a repository is likely to offer, by format (a 4-bit quantization for GGUF and ONNX).</summary>
    private const double QuantizedBytesPerParameter = 0.55;

    /// <summary>Bytes per parameter of a bf16 checkpoint, the smallest a safetensors repository without a quantization offers.</summary>
    private const double CheckpointBytesPerParameter = 2.0;

    /// <summary>Bytes per parameter of an 8-bit checkpoint.</summary>
    private const double Fp8BytesPerParameter = 1.05;

    [GeneratedRegex("[-_./ ]+")]
    private static partial Regex Separators();

    /// <summary>The quantization families the filter offers (matched at the start of a name segment: <c>Q4</c> matches <c>Q4_K_M</c> and <c>q4f16</c>).</summary>
    public static IReadOnlyList<string> Quantizations { get; } = ["Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "IQ", "F16", "BF16", "FP8", "MXFP4", "NVFP4"];

    /// <summary>The Hub query for <paramref name="settings"/> and the search text.</summary>
    /// <param name="settings">The filters.</param>
    /// <param name="text">The search text.</param>
    /// <param name="limit">Maximum number of results.</param>
    public static HubSearchQuery ToQuery(DiscoverFilterSettings settings, string text, int limit) => new()
    {
        Text = text,
        Sort = settings.Sort,
        Format = settings.Format,
        Task = settings.Task,
        License = settings.License,
        Limit = limit,
    };

    /// <summary>Whether a result passes the filters the Hub does not apply.</summary>
    /// <param name="model">The search result.</param>
    /// <param name="settings">The filters.</param>
    /// <param name="memoryBytes">The memory a model can use on this machine (<see cref="Capacity"/>), or null when unknown (nothing is hidden).</param>
    public static bool Matches(HubModelSummary model, DiscoverFilterSettings settings, long? memoryBytes)
    {
        if (settings.Gated == GatedFilter.Hide && model.Gated || settings.Gated == GatedFilter.Only && !model.Gated)
        {
            return false;
        }

        if (settings.Quantization is { Length: > 0 } quantization && !OffersQuantization(model, quantization))
        {
            return false;
        }

        if (settings.Size != ModelSizeFilter.Any && model.Parameters is { } parameters && !InRange(parameters, settings.Size))
        {
            return false;
        }

        if (settings.FitsThisMachine && memoryBytes is { } memory && SmallestDownload(model) is { } smallest && smallest > memory)
        {
            return false;
        }

        return true;
    }

    /// <summary>The memory a model can use here: the larger of all GPUs together and the system memory (CPU offload runs from it).</summary>
    /// <param name="gpuBytes">Total memory of every GPU.</param>
    /// <param name="systemBytes">Total system memory.</param>
    public static long Capacity(long gpuBytes, long systemBytes) => Math.Max(gpuBytes, systemBytes);

    /// <summary>The approximate size of the smallest download of a repository, from its parameter count; null when the Hub reports none.</summary>
    /// <param name="model">The search result.</param>
    public static long? SmallestDownload(HubModelSummary model)
    {
        if (model.Parameters is not { } parameters)
        {
            return null;
        }

        double perParameter = double.MaxValue;
        if (model.Formats.Contains(ModelFormat.Gguf) || model.Formats.Contains(ModelFormat.Onnx))
        {
            perParameter = QuantizedBytesPerParameter;
        }

        if (model.Formats.Contains(ModelFormat.Safetensors))
        {
            perParameter = Math.Min(perParameter, QuantizationBytes(model.QuantizationMethod));
        }

        return perParameter == double.MaxValue ? null : (long)(parameters * perParameter);
    }

    /// <summary>The label of a size range.</summary>
    /// <param name="size">The range.</param>
    public static string Describe(ModelSizeFilter size) => size switch
    {
        ModelSizeFilter.Under1B => "Under 1B parameters",
        ModelSizeFilter.From1To4B => "1B to 4B",
        ModelSizeFilter.From4To14B => "4B to 14B",
        ModelSizeFilter.From14To40B => "14B to 40B",
        ModelSizeFilter.Over40B => "Over 40B",
        _ => "Any size",
    };

    private static double QuantizationBytes(string? method)
        => method is null ? CheckpointBytesPerParameter : method.Contains("fp8", StringComparison.OrdinalIgnoreCase) ? Fp8BytesPerParameter : QuantizedBytesPerParameter;

    private static bool InRange(long parameters, ModelSizeFilter size)
    {
        const long billion = 1_000_000_000;
        return size switch
        {
            ModelSizeFilter.Under1B => parameters < billion,
            ModelSizeFilter.From1To4B => parameters is >= billion and < 4 * billion,
            ModelSizeFilter.From4To14B => parameters is >= 4 * billion and < 14 * billion,
            ModelSizeFilter.From14To40B => parameters is >= 14 * billion and < 40 * billion,
            ModelSizeFilter.Over40B => parameters >= 40 * billion,
            _ => true,
        };
    }

    /// <summary>A variant label (or the checkpoint's quantization method) in the family: a name segment that starts with it.</summary>
    private static bool OffersQuantization(HubModelSummary model, string family)
    {
        if (model.QuantizationMethod is { } method && method.StartsWith(family, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        foreach (var label in model.Variants)
        {
            if (Separators().Split(label).Any(segment => segment.StartsWith(family, StringComparison.OrdinalIgnoreCase)))
            {
                return true;
            }
        }

        return false;
    }
}
