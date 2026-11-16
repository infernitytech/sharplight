using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Completes a transformer-only diffusion checkpoint (GGUF / single files distributed without their text encoders and VAE) with
/// companion files loaded as auxiliary sources: each accepted role names the companion architecture it takes and where its canonical
/// weights and metadata go in the family's vocabulary (a companion replaces what the primary source already has; keys the family fixes,
/// such as the VAE's latent scaling, stay the primary's). The merged source is built by the family's adapter, so the model is rebuilt for
/// other sizes and saved as one package like a complete checkpoint.
/// </summary>
public abstract class DiffusionCompanionAdapter : ICompositeModelAdapter
{
    /// <summary>Canonical architecture of a stand-alone CLIP text encoder source (weights without prefix, metadata <c>clip_text_encoder.*</c>).</summary>
    public const string ClipTextArchitecture = "clip_text_encoder";

    /// <summary>Canonical architecture of a stand-alone KL autoencoder source (weights <c>vae.*</c>, metadata <c>kl_autoencoder.vae.*</c>).</summary>
    public const string AutoencoderArchitecture = "kl_autoencoder";

    /// <summary>Canonical architecture of a T5 encoder source (the encoder of an <c>encoder_decoder</c> model).</summary>
    public const string T5Architecture = "encoder_decoder";

    /// <summary>Architecture id of stand-alone causal 3-D (Wan 2.1-layout) VAE sources.</summary>
    public const string WanAutoencoderArchitecture = "wan_autoencoder";

    private const string TokenizerKeys = "tokenizer.";

    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <summary>The architecture of the transformer-only primary sources.</summary>
    protected abstract string Architecture { get; }

    /// <summary>The companion slots by auxiliary role.</summary>
    protected abstract IReadOnlyDictionary<string, CompanionSlot> Slots { get; }

    /// <summary>Builds the merged (complete) source.</summary>
    /// <param name="merged">The primary source with the companions merged in.</param>
    /// <param name="options">The adapter options.</param>
    protected abstract ModelDefinition BuildMerged(ModelSource merged, AdapterOptions options);

    /// <inheritdoc/>
    public bool CanLoad(ModelSourceSet sources)
    {
        ArgumentNullException.ThrowIfNull(sources);
        if (sources.Primary.Architecture != Architecture || sources.Auxiliary.Count == 0)
        {
            return false;
        }

        return sources.Auxiliary.All(pair => Slots.TryGetValue(pair.Key, out var slot) && slot.Accepts(pair.Value.Architecture));
    }

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options, ArchitectureCatalog architectures)
    {
        ArgumentNullException.ThrowIfNull(sources);
        var primary = sources.Primary;
        string prefix = Architecture + ".";
        var metadata = new Dictionary<string, object>(primary.Metadata, StringComparer.Ordinal);
        var weights = new Dictionary<string, WeightInfo>(primary.Weights, StringComparer.Ordinal);
        var resources = new List<IDisposable>(primary.OwnedResources);
        foreach (var (role, source) in sources.Auxiliary)
        {
            var slot = Slots[role];
            foreach (string name in weights.Keys.Where(n => n.StartsWith(slot.WeightPrefix, StringComparison.Ordinal)).ToList())
            {
                weights.Remove(name);
            }

            foreach (var (name, weight) in source.Weights)
            {
                if (slot.Rename(name) is { } canonical)
                {
                    weights[canonical] = weight.As(canonical);
                }
            }

            string target = prefix + slot.MetadataKey + ".";
            var kept = slot.FamilyKeys.Where(k => metadata.ContainsKey(target + k)).ToDictionary(k => k, k => metadata[target + k]);
            CopyMetadata(source, metadata, slot.SourceMetadataPrefix, target);
            foreach (var (key, value) in kept)
            {
                metadata[target + key] = value;
            }

            if (slot.ArchitectureKey is { } architectureKey)
            {
                metadata[target + architectureKey] = source.Architecture;
            }

            if (slot.TokenizerMetadataPrefix is { } tokenizer)
            {
                CopyMetadata(source, metadata, TokenizerKeys, tokenizer.Length == 0 ? TokenizerKeys : prefix + tokenizer + TokenizerKeys);
            }

            resources.AddRange(source.OwnedResources);
        }

        var merged = new ModelSource
        {
            Format = primary.Format,
            Architecture = Architecture,
            Metadata = metadata,
            Weights = weights,
            OwnedResources = resources,
            Path = primary.Path,
            FormatData = primary.FormatData,
            CompilationCaches = primary.CompilationCaches,
        };
        return BuildMerged(merged, options);
    }

    /// <summary>A CLIP text encoder slot: weights under <paramref name="weightPrefix"/>, metadata under <c>{arch}.{metadataKey}.</c>.</summary>
    /// <param name="weightPrefix">Canonical weight prefix (<c>text_encoder.</c>).</param>
    /// <param name="metadataKey">Metadata component key (<c>text_encoder</c>).</param>
    /// <param name="tokenizerMetadataPrefix">Where its tokenizer goes ("" = the canonical <c>tokenizer.*</c> keys), or null to drop it.</param>
    /// <param name="familyKeys">Keys under the component the family fixes (e.g. <c>clip_skip</c>).</param>
    protected static CompanionSlot Clip(string weightPrefix, string metadataKey, string? tokenizerMetadataPrefix, params string[] familyKeys)
        => new([ClipTextArchitecture], weightPrefix, name => weightPrefix + name, ClipTextArchitecture + ".", metadataKey, tokenizerMetadataPrefix, familyKeys);

    /// <summary>A T5 encoder slot: the encoder weights under <paramref name="weightPrefix"/>, its tokenizer under <c>{arch}.{tokenizerMetadataPrefix}tokenizer.*</c>.</summary>
    /// <param name="weightPrefix">Canonical weight prefix (<c>text_encoder_2.</c>).</param>
    /// <param name="metadataKey">Metadata component key (<c>text_encoder_2</c>).</param>
    /// <param name="tokenizerMetadataPrefix">Tokenizer metadata prefix after the architecture (<c>tokenizer_2.</c>).</param>
    protected static CompanionSlot T5(string weightPrefix, string metadataKey, string tokenizerMetadataPrefix)
        => new([T5Architecture], weightPrefix,
            name => name.StartsWith("encoder.", StringComparison.Ordinal) || name == "token_embedding.weight" ? weightPrefix + name : null,
            T5Architecture + ".", metadataKey, tokenizerMetadataPrefix, ["prompt_length"]);

    /// <summary>
    /// A language model kept whole as the text encoder (Z-Image's Qwen3): every weight under <paramref name="weightPrefix"/>, every metadata key
    /// under <c>{arch}.{metadataKey}.</c> with its architecture as <c>architecture</c>, its tokenizer under the canonical keys.
    /// </summary>
    /// <param name="weightPrefix">Canonical weight prefix (<c>text_encoder.</c>).</param>
    /// <param name="metadataKey">Metadata component key (<c>text_encoder_source</c>).</param>
    /// <param name="architectures">The language-model source architectures accepted (<c>decoder</c>).</param>
    protected static CompanionSlot LanguageModel(string weightPrefix, string metadataKey, params string[] architectures)
        => new(architectures, weightPrefix, name => weightPrefix + name, "", metadataKey, "", [], "architecture");

    /// <summary>The VAE slot (weights <c>vae.*</c>, the family's latent scaling kept).</summary>
    protected static CompanionSlot Vae()
        => new([AutoencoderArchitecture], "vae.", name => name.StartsWith("vae.", StringComparison.Ordinal) ? name : null,
            AutoencoderArchitecture + ".vae.", "vae", null, ["scaling_factor", "shift_factor", "latent_rgb_factors"]);

    /// <summary>
    /// The slot of a causal 3-D VAE file (Qwen-Image's; architecture <c>wan_autoencoder</c>): its <c>vae.*</c> weights and its
    /// <c>wan_autoencoder.vae.*</c> keys under the family's <c>vae.</c>.
    /// </summary>
    protected static CompanionSlot CausalVae()
        => new([WanAutoencoderArchitecture], "vae.", name => name.StartsWith("vae.", StringComparison.Ordinal) ? name : null,
            WanAutoencoderArchitecture + ".vae.", "vae", null, []);

    /// <summary>Replaces the metadata keys under <paramref name="to"/> with the source's keys under <paramref name="from"/> (prefix replaced).</summary>
    private static void CopyMetadata(ModelSource source, Dictionary<string, object> metadata, string from, string to)
    {
        foreach (string key in metadata.Keys.Where(k => k.StartsWith(to, StringComparison.Ordinal)).ToList())
        {
            metadata.Remove(key);
        }

        foreach (var (key, value) in source.Metadata)
        {
            if (key.StartsWith(from, StringComparison.Ordinal))
            {
                metadata[to + key[from.Length..]] = value;
            }
        }
    }
}
