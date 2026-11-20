using System.Globalization;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.FeedTool;

/// <summary>Opens a built package with the engine's reader (no weights are loaded) and reads what the manifest needs.</summary>
internal static class PackageInspector
{
    /// <summary>The context length the default memory figure is given for (the engine's default load context).</summary>
    public const int DefaultContextTokens = PrecompileOptions.DefaultContextLength;

    /// <summary>Prefill batch the activation estimate assumes (tokens).</summary>
    private const int PrefillTokens = 512;

    private static readonly string[] ContextKeys = ["context_length", "max_position_embeddings", "n_ctx_train", "max_context", "n_text_ctx"];

    /// <summary>Inspects <paramref name="path"/>.</summary>
    /// <exception cref="InvalidDataException">The file is not a readable package.</exception>
    public static PackageInspection Inspect(string path)
    {
        var info = ModelPackageReader.Open(path);
        var definition = info.Definition;
        var properties = definition.Properties.Concat(definition.Metadata).ToList();
        var parameters = info.Weights.Sum(weight => weight.Shape.Elements());
        var template = properties.Any(property => property.Key.Contains("chat_template", StringComparison.OrdinalIgnoreCase));
        var contextLength = ContextLength(properties);
        var types = info.Weights.GroupBy(weight => weight.Type.Name, StringComparer.OrdinalIgnoreCase)
            .Select(group => KeyValuePair.Create(group.Key.ToLowerInvariant(), group.Sum(weight => weight.ByteSize)))
            .OrderByDescending(pair => pair.Value)
            .ThenBy(pair => pair.Key, StringComparer.Ordinal)
            .ToList();
        return new PackageInspection(
            string.IsNullOrWhiteSpace(definition.Name) ? Path.GetFileNameWithoutExtension(path) : definition.Name,
            definition.HasDescription ? definition.Description.Kind : "text-generation",
            string.IsNullOrWhiteSpace(definition.Architecture) ? null : definition.Architecture,
            contextLength,
            parameters,
            info.WeightBytes,
            info.Version,
            template)
        {
            WeightTypes = types,
            Context = definition.Graphs.Any(graph => graph.Name == WellKnown.DecodeGraph) ? MeasureDecoder(path, contextLength ?? DefaultContextTokens) : null,
        };
    }

    // Opens the full definition (weights stay in the file) to size the decode graph's state.
    private static ContextMemory? MeasureDecoder(string path, int maxTokens)
    {
        var source = new ModelPackageImporter().Open(path);
        try
        {
            return source.Definition is { } full && full.Graphs.TryGetValue(WellKnown.DecodeGraph, out var decode) ? Measure(decode, maxTokens) : null;
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }

    /// <summary>
    /// Measures the per-slot state of <paramref name="decode"/>: exactly at the default and the maximum context, and the growth per token
    /// between half the maximum and the maximum (where sliding windows are saturated), so state + per-token · context is exact at long
    /// contexts and an upper bound at short ones.
    /// </summary>
    /// <param name="decode">The decode graph.</param>
    /// <param name="maxTokens">The model's maximum context.</param>
    /// <returns>The figures, or <see langword="null"/> when the graph has no state.</returns>
    public static ContextMemory? Measure(ComputeGraph decode, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(decode);
        int max = Math.Max(maxTokens, 2);
        int defaultTokens = Math.Min(DefaultContextTokens, max);
        int half = max / 2;
        long atMax = StateBytes(decode, max);
        if (atMax <= 0)
        {
            return null;
        }

        long atHalf = StateBytes(decode, half);
        long perToken = Math.Max(0, (atMax - atHalf) / Math.Max(1, max - half));
        long fixedBytes = Math.Max(0, atMax - (perToken * max));
        long activations = PlacementPlanner.Estimate(decode, new CompileOptions { MaxTokens = PrefillTokens, ContextLength = defaultTokens }).ActivationBytes;
        return new ContextMemory(perToken, fixedBytes, defaultTokens, StateBytes(decode, defaultTokens), max, atMax, activations);
    }

    private static long StateBytes(ComputeGraph decode, int context) =>
        PlacementPlanner.Estimate(decode, new CompileOptions { MaxTokens = 1, ContextLength = context }).StateBytes;

    private static int? ContextLength(IEnumerable<KeyValuePair<string, object>> properties)
    {
        foreach (var (key, value) in properties)
        {
            if (ContextKeys.Any(candidate => key.EndsWith(candidate, StringComparison.OrdinalIgnoreCase))
                && long.TryParse(Convert.ToString(value, CultureInfo.InvariantCulture), NumberStyles.Integer, CultureInfo.InvariantCulture, out var length)
                && length is > 0 and <= int.MaxValue)
            {
                return (int)length;
            }
        }

        return null;
    }
}
