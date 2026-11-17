using System.Globalization;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Loads a transformer-only HunyuanVideo checkpoint (GGUF or original-name safetensors) with its companions: the language model with its
/// tokenizer and CLIP-L with its tokenizer (<see cref="AuxiliaryRoles.TextEncoder"/>) and the VAE (<see cref="AuxiliaryRoles.Vae"/>), taken
/// from complete or configuration-and-encoder <c>hunyuan_video</c> sources (a HunyuanVideo diffusers folder serves both roles). The parts are
/// merged into the canonical source, which <see cref="HunyuanVideoAdapter"/> builds.
/// </summary>
public sealed class HunyuanVideoCompanionAdapter : ICompositeModelAdapter
{
    private const string Prefix = HunyuanVideoAdapter.Architecture + ".";

    /// <inheritdoc/>
    public string Name => "HunyuanVideo with companion sources";

    /// <inheritdoc/>
    public bool CanLoad(ModelSourceSet sources)
    {
        ArgumentNullException.ThrowIfNull(sources);
        return sources.Primary.Architecture == HunyuanVideoAdapter.Architecture && sources.Auxiliary.Count > 0
            && sources.Auxiliary.All(pair => pair.Key is AuxiliaryRoles.TextEncoder or AuxiliaryRoles.Vae && pair.Value.Architecture == HunyuanVideoAdapter.Architecture);
    }

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options, ArchitectureCatalog architectures)
    {
        ArgumentNullException.ThrowIfNull(sources);
        var primary = sources.Primary;
        var metadata = new Dictionary<string, object>(primary.Metadata, StringComparer.Ordinal);
        var weights = new Dictionary<string, WeightInfo>(primary.Weights, StringComparer.Ordinal);
        var resources = new List<IDisposable>(primary.OwnedResources);
        foreach (var (role, source) in sources.Auxiliary)
        {
            if (role == AuxiliaryRoles.TextEncoder)
            {
                Take(source, weights, metadata, ["text_encoder.", "text_encoder_2."],
                    [Prefix + "text_encoder_source.", Prefix + "text_encoder.", Prefix + "text_encoder_2.", Prefix + "tokenizer_2.", "tokenizer."]);
                TakeFamily(source, metadata);
            }
            else
            {
                Take(source, weights, metadata, ["vae."], [Prefix + "vae."]);
            }

            resources.AddRange(source.OwnedResources);
        }

        var merged = new ModelSource
        {
            Format = primary.Format,
            Architecture = HunyuanVideoAdapter.Architecture,
            Metadata = metadata,
            Weights = weights,
            OwnedResources = resources,
            Path = primary.Path,
            FormatData = primary.FormatData,
            CompilationCaches = primary.CompilationCaches,
        };
        return new HunyuanVideoAdapter().Build(new ModelSourceSet { Primary = merged }, options);
    }

    /// <summary>
    /// A transformer-only file records neither its head layout (configured as 128-wide heads) nor its RoPE axes, scheduler or defaults; a
    /// companion folder of the same family (equal width) does.
    /// </summary>
    private static void TakeFamily(ModelSource source, Dictionary<string, object> metadata)
    {
        const string hidden = Prefix + "transformer.hidden_size";
        if (!source.Metadata.TryGetValue(hidden, out var width) || !metadata.TryGetValue(hidden, out var own)
            || Convert.ToInt64(own, CultureInfo.InvariantCulture) != Convert.ToInt64(width, CultureInfo.InvariantCulture))
        {
            return;
        }

        foreach (var (key, value) in source.Metadata)
        {
            if (key is Prefix + "transformer.head_count" or Prefix + "transformer.head_dim" or Prefix + "transformer.rope_axes_dim" or Prefix + "transformer.rope_theta"
                || key.StartsWith(Prefix + "default_", StringComparison.Ordinal) || key.StartsWith(Prefix + "scheduler.", StringComparison.Ordinal))
            {
                metadata[key] = value;
            }
        }
    }

    /// <summary>Replaces the weights under <paramref name="weightPrefixes"/> and the metadata under <paramref name="metadataPrefixes"/> with the companion's.</summary>
    private static void Take(ModelSource source, Dictionary<string, WeightInfo> weights, Dictionary<string, object> metadata, IReadOnlyList<string> weightPrefixes,
        IReadOnlyList<string> metadataPrefixes)
    {
        foreach (string prefix in weightPrefixes)
        {
            foreach (string name in weights.Keys.Where(n => n.StartsWith(prefix, StringComparison.Ordinal)).ToList())
            {
                weights.Remove(name);
            }

            foreach (var (name, weight) in source.Weights)
            {
                if (name.StartsWith(prefix, StringComparison.Ordinal))
                {
                    weights[name] = weight;
                }
            }
        }

        foreach (string prefix in metadataPrefixes)
        {
            foreach (string key in metadata.Keys.Where(k => k.StartsWith(prefix, StringComparison.Ordinal)).ToList())
            {
                metadata.Remove(key);
            }

            foreach (var (key, value) in source.Metadata)
            {
                if (key.StartsWith(prefix, StringComparison.Ordinal))
                {
                    metadata[key] = value;
                }
            }
        }
    }
}
