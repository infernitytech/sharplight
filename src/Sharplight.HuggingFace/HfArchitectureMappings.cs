using System.Collections.Concurrent;

namespace Sharplight.HuggingFace;

/// <summary>Registry of <see cref="IHfArchitectureMapping"/>s keyed by HF <c>model_type</c> (case-insensitive).</summary>
internal static class HfArchitectureMappings
{
    private static readonly ConcurrentDictionary<string, IHfArchitectureMapping> s_byModelType = new(StringComparer.OrdinalIgnoreCase);

    static HfArchitectureMappings()
    {
        Register(new Qwen35HfMapping());
        Register(new NemotronHHfMapping());
        Register(new GraniteHybridHfMapping());
        Register(new FalconH1HfMapping());
        Register(new Mamba2HfMapping());
        Register(new Lfm2HfMapping());
        Register(new MambaHfMapping());
        Register(new DecoderHfMapping());
        Register(new EncoderHfMapping());
    }

    /// <summary>Registers <paramref name="mapping"/> for each of its model types; later registrations win.</summary>
    public static void Register(IHfArchitectureMapping mapping)
    {
        foreach (var modelType in mapping.ModelTypes)
        {
            s_byModelType[modelType] = mapping;
        }
    }

    /// <summary>Every registered mapping once.</summary>
    public static IEnumerable<IHfArchitectureMapping> All => s_byModelType.Values.Distinct();

    /// <summary>The mapping for a <c>model_type</c>, or null.</summary>
    public static IHfArchitectureMapping? Find(string? modelType)
        => modelType != null && s_byModelType.TryGetValue(modelType, out var m) ? m : null;
}
