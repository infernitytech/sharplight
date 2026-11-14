using System.Collections.Concurrent;

namespace Sharplight.Gguf;

/// <summary>Registry of <see cref="IGgufArchitectureMapping"/>s keyed by GGUF architecture id (case-insensitive).</summary>
internal static class GgufArchitectureMappings
{
    private static readonly ConcurrentDictionary<string, IGgufArchitectureMapping> s_byArch = new(StringComparer.OrdinalIgnoreCase);

    static GgufArchitectureMappings()
    {
        Register(new Qwen35GgufMapping());
        Register(new Qwen3VlGgufMapping());
        Register(new NemotronHGgufMapping());
        Register(new HybridGgufMapping());
        Register(new MiniMax01GgufMapping());
        Register(new Rwkv7GgufMapping());
        Register(new Rwkv6GgufMapping());
        Register(new LatentDecoderGgufMapping());
        Register(new Eagle3GgufMapping());
        Register(new ClipGgufMapping());
        Register(new DecoderGgufMapping());
        Register(new EncoderGgufMapping());
        Register(new T5GgufMapping());
        foreach (var family in DecoderFamilies())
        {
            Register(family);
        }
    }

    /// <summary>The llama.cpp decoder-only families that map onto the canonical decoder through <see cref="GgufDecoderFamilyMapping"/>.</summary>
    private static IEnumerable<IGgufArchitectureMapping> DecoderFamilies()
    {
        yield return new SmolLm3GgufMapping();
        yield return new Olmo2GgufMapping();
        yield return new ChameleonGgufMapping();
        yield return new Glm4GgufMapping();
        yield return new Ernie45GgufMapping();
        yield return new HunyuanDenseGgufMapping();
        yield return new ExaoneGgufMapping();
        yield return new Exaone4GgufMapping();
        yield return new SeedOssGgufMapping();
        yield return new InternLm2GgufMapping();
        yield return new Qwen1GgufMapping();
        yield return new XverseGgufMapping();
        yield return new Gemma1GgufMapping();
        yield return new OpenElmGgufMapping();
        yield return new GraniteGgufMapping();
        yield return new MiniCpmGgufMapping();
        yield return new Phi2GgufMapping();
        yield return new Starcoder2GgufMapping();
        yield return new CommandRGgufMapping();
        yield return new FalconGgufMapping();
        yield return new GptNeoXGgufMapping();
        yield return new Gpt2GgufMapping();
        yield return new OlmoGgufMapping();
        yield return new NemotronGgufMapping();
        yield return new StableLmGgufMapping();
        yield return new ApertusGgufMapping();
        yield return new ArceeGgufMapping();
        yield return new Glm4MoeGgufMapping();
        yield return new Ernie45MoeGgufMapping();
        yield return new RoutedMoeGgufMapping();
        yield return new MiniMaxM3GgufMapping();
        yield return new Qwen2MoeGgufMapping();
        yield return new RefactGgufMapping();
        yield return new JaisGgufMapping();
        yield return new OrionGgufMapping();
        yield return new CodeShellGgufMapping();
        yield return new PlamoGgufMapping();
        yield return new Plamo3GgufMapping();
        yield return new PanguEmbeddedGgufMapping();
        yield return new MaincoderGgufMapping();
        yield return new Spark25GgufMapping();
        yield return new BaichuanGgufMapping();
        yield return new Jais2GgufMapping();
        yield return new GraniteSwaGgufMapping();
        yield return new MellumGgufMapping();
        yield return new DbrxGgufMapping();
        yield return new PhimoeGgufMapping();
        yield return new MaionCoderGgufMapping();
        yield return new BeamGgufMapping();
        yield return new MuseSparkGgufMapping();
        yield return new DeepseekGgufMapping();
        yield return new BailingMoeGgufMapping();
        yield return new Cohere2MoeGgufMapping();
        yield return new HunyuanMoeGgufMapping();
        yield return new ExaoneMoeGgufMapping();
        yield return new SmallThinkerGgufMapping();
        yield return new GptOssGgufMapping();
        yield return new Llama4GgufMapping();
        yield return new Mistral3GgufMapping();
        yield return new BitnetGgufMapping();
        yield return new AfmoeGgufMapping();
        yield return new Mimo2GgufMapping();
        yield return new LagunaGgufMapping();
        yield return new BloomGgufMapping();
        yield return new MptGgufMapping();
        yield return new TalkieGgufMapping();
        yield return new NanbeigeGgufMapping();
        yield return new ArcticGgufMapping();
        yield return new DeciGgufMapping();
        yield return new GrokGgufMapping();
        yield return new MapleGgufMapping();
        yield return new Step35GgufMapping();
        yield return new Kolibri1GgufMapping();
        yield return new RicursiveGgufMapping();
        yield return new EuroBertGgufMapping();
        yield return new DreamGgufMapping();
        yield return new LladaGgufMapping();
        yield return new NeoBertGgufMapping();
        yield return new LlamaEmbedGgufMapping();
        yield return new GemmaEmbeddingGgufMapping();
        yield return new ModernBertGgufMapping();
    }

    /// <summary>Registers <paramref name="mapping"/> for each of its GGUF architectures; later registrations win.</summary>
    public static void Register(IGgufArchitectureMapping mapping)
    {
        foreach (var architecture in mapping.GgufArchitectures)
        {
            s_byArch[architecture] = mapping;
        }
    }

    /// <summary>Every registered mapping once.</summary>
    public static IEnumerable<IGgufArchitectureMapping> All => s_byArch.Values.Distinct();

    /// <summary>The mapping for a <c>general.architecture</c> value, or null.</summary>
    public static IGgufArchitectureMapping? Find(string? ggufArchitecture)
        => ggufArchitecture != null && s_byArch.TryGetValue(ggufArchitecture, out var m) ? m : null;
}
