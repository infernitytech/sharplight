using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Loads a transformer-only LTX-Video checkpoint (GGUF or original-name safetensors) with its companions: the T5 encoder with its tokenizer
/// (<see cref="AuxiliaryRoles.TextEncoder"/>, an LTX-Video diffusers folder) and the VAE (<see cref="AuxiliaryRoles.Vae"/>, a stand-alone
/// VAE file or a folder), taken from <c>ltx</c> sources and merged into the canonical source, which <see cref="LtxAdapter"/> builds.
/// </summary>
public sealed class LtxCompanionAdapter : ICompositeModelAdapter
{
    private const string Prefix = LtxAdapter.Architecture + ".";

    /// <inheritdoc/>
    public string Name => "LTX-Video with companion sources";

    /// <inheritdoc/>
    public bool CanLoad(ModelSourceSet sources)
    {
        ArgumentNullException.ThrowIfNull(sources);
        return sources.Primary.Architecture == LtxAdapter.Architecture && sources.Auxiliary.Count > 0
            && sources.Auxiliary.All(pair => pair.Key is AuxiliaryRoles.TextEncoder or AuxiliaryRoles.Vae && pair.Value.Architecture == LtxAdapter.Architecture);
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
                Take(source, weights, metadata, "text_encoder.", [Prefix + "text_encoder.", "tokenizer."]);
            }
            else
            {
                Take(source, weights, metadata, "vae.", [Prefix + "vae."]);
            }

            resources.AddRange(source.OwnedResources);
        }

        var merged = new ModelSource
        {
            Format = primary.Format,
            Architecture = LtxAdapter.Architecture,
            Metadata = metadata,
            Weights = weights,
            OwnedResources = resources,
            Path = primary.Path,
            FormatData = primary.FormatData,
            CompilationCaches = primary.CompilationCaches,
        };
        return new LtxAdapter().Build(new ModelSourceSet { Primary = merged }, options);
    }

    /// <summary>Replaces the weights under <paramref name="weightPrefix"/> and the metadata under <paramref name="metadataPrefixes"/> with the companion's.</summary>
    private static void Take(ModelSource source, Dictionary<string, WeightInfo> weights, Dictionary<string, object> metadata, string weightPrefix,
        IReadOnlyList<string> metadataPrefixes)
    {
        foreach (string name in weights.Keys.Where(n => n.StartsWith(weightPrefix, StringComparison.Ordinal)).ToList())
        {
            weights.Remove(name);
        }

        foreach (var (name, weight) in source.Weights)
        {
            if (name.StartsWith(weightPrefix, StringComparison.Ordinal))
            {
                weights[name] = weight;
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
