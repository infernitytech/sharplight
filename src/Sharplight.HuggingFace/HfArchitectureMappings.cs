using System.Collections.Concurrent;

namespace Sharplight.HuggingFace;

/// <summary>Registry of <see cref="IHfArchitectureMapping"/>s keyed by HF <c>model_type</c> (case-insensitive).</summary>
internal static class HfArchitectureMappings
{
    private static readonly ConcurrentDictionary<string, IHfArchitectureMapping> s_byModelType = new(StringComparer.OrdinalIgnoreCase);

    static HfArchitectureMappings()
    {
        Register(new Qwen35HfMapping());
        Register(new Qwen3VlHfMapping());
        Register(new NemotronHHfMapping());
        Register(new GraniteHybridHfMapping());
        Register(new FalconH1HfMapping());
        Register(new Mamba2HfMapping());
        Register(new Lfm2HfMapping());
        Register(new MambaHfMapping());
        Register(new LatentDecoderHfMapping());
        Register(new DecoderHfMapping());
        Register(new EncoderHfMapping());
        Register(new T5HfMapping());
        foreach (var family in DecoderFamilies())
        {
            Register(family);
        }
    }

    /// <summary>The decoder-only families that map onto the canonical decoder through <see cref="HfDecoderFamilyMapping"/>.</summary>
    private static IEnumerable<IHfArchitectureMapping> DecoderFamilies()
    {
        yield return new MixtralHfMapping();
        yield return new AfmoeHfMapping();
        yield return new PhimoeHfMapping();
        yield return new Kolibri1HfMapping();
        yield return new MiMoV2FlashHfMapping();
        yield return new LagunaHfMapping();
        yield return new BloomHfMapping();
        yield return new MptHfMapping();
        yield return new SmolLm3HfMapping();
        yield return new Olmo2HfMapping();
        yield return new Glm4HfMapping();
        yield return new Ernie45HfMapping();
        yield return new HunyuanDenseHfMapping();
        yield return new Glm4MoeHfMapping();
        yield return new Ernie45MoeHfMapping();
        yield return new HunyuanMoeHfMapping();
        yield return new MiniMaxM2HfMapping();
        yield return new BailingMoeV2HfMapping();
        yield return new GptOssHfMapping();
        yield return new Llama4HfMapping();
        yield return new Dots1HfMapping();
        yield return new Qwen2MoeHfMapping();
        yield return new Jais2HfMapping();
        yield return new GraniteSwaHfMapping();
        yield return new MellumHfMapping();
        yield return new DbrxHfMapping();
        yield return new Cohere2MoeHfMapping();
        yield return new HyV3HfMapping();
        yield return new ExaoneMoeHfMapping();
        yield return new BitnetHfMapping();
        yield return new Ministral3HfMapping();
        yield return new ExaoneHfMapping();
        yield return new Exaone4HfMapping();
        yield return new SeedOssHfMapping();
        yield return new InternLm2HfMapping();
        yield return new GraniteHfMapping();
        yield return new MiniCpmHfMapping();
        yield return new PhiHfMapping();
        yield return new Starcoder2HfMapping();
        yield return new CohereHfMapping();
        yield return new FalconHfMapping();
        yield return new GptNeoXHfMapping();
        yield return new GptJHfMapping();
        yield return new Gpt2HfMapping();
        yield return new OlmoHfMapping();
        yield return new NemotronDenseHfMapping();
        yield return new StableLmHfMapping();
        yield return new ApertusHfMapping();
        yield return new ArceeHfMapping();
        yield return new Gemma1HfMapping();
        yield return new EuroBertHfMapping();
        yield return new ModernBertHfMapping();
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
