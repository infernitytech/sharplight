using System.Globalization;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Loads a transformer-only Wan checkpoint (GGUF or original-name safetensors) with its companions: the text encoder and its
/// tokenizer (<see cref="AuxiliaryRoles.TextEncoder"/>), the VAE (<see cref="AuxiliaryRoles.Vae"/>) and for Wan 2.1 I2V / FLF2V the
/// CLIP image encoder (<see cref="AuxiliaryRoles.ImageEncoder"/>; a folder with only the configurations and the image encoder serves), each taken from a complete
/// <c>wan</c> source of the same family (typically the model's diffusers folder; one folder may serve both roles). The parts are merged
/// into the canonical source, which <see cref="WanAdapter"/> builds, so the model is rebuilt for other sizes and saved as one
/// <c>.slm</c> like a complete checkpoint.
/// </summary>
public sealed class WanCompanionAdapter : ICompositeModelAdapter
{
    private const string Prefix = WanAdapter.Architecture + ".";
    private const string TokenizerKeys = "tokenizer.";

    private static readonly HashSet<string> s_familyKeys =
    [
        Prefix + "transformer.head_count", Prefix + "transformer.token_timesteps", Prefix + "first_frame_conditioning", Prefix + "frames_per_second",
        Prefix + "negative_prompt", Prefix + "last_frame_conditioning",
    ];

    /// <inheritdoc/>
    public string Name => "Wan with companion sources";

    /// <inheritdoc/>
    public bool CanLoad(ModelSourceSet sources)
    {
        ArgumentNullException.ThrowIfNull(sources);
        return sources.Primary.Architecture == WanAdapter.Architecture && sources.Auxiliary.Count > 0
            && sources.Auxiliary.All(pair => pair.Key is AuxiliaryRoles.TextEncoder or AuxiliaryRoles.Vae or AuxiliaryRoles.SecondTransformer or AuxiliaryRoles.ImageEncoder
                && pair.Value.Architecture == WanAdapter.Architecture);
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
            if (role == AuxiliaryRoles.SecondTransformer)
            {
                AddSecondTransformer(source, weights, metadata);
                resources.AddRange(source.OwnedResources);
                continue;
            }

            if (role == AuxiliaryRoles.TextEncoder)
            {
                Take(source, weights, metadata, "text_encoder.", [Prefix + "text_encoder.", TokenizerKeys]);
            }
            else if (role == AuxiliaryRoles.ImageEncoder)
            {
                Take(source, weights, metadata, "image_encoder.", [Prefix + "image_encoder."]);
            }
            else
            {
                Take(source, weights, metadata, "vae.", [Prefix + "vae."]);
            }

            // A transformer-only file records neither its head count (configured as 128-wide heads) nor whether it takes per-token
            // timesteps (inferred from the channel count), nor the pipeline's defaults and scheduler; a companion of the same family
            // (equal width) does.
            const string hidden = Prefix + "transformer.hidden_size";
            if (source.Metadata.TryGetValue(hidden, out var width) && metadata.TryGetValue(hidden, out var own)
                && Convert.ToInt64(own, CultureInfo.InvariantCulture) == Convert.ToInt64(width, CultureInfo.InvariantCulture))
            {
                foreach (var (key, value) in source.Metadata)
                {
                    if (s_familyKeys.Contains(key) || key.StartsWith(Prefix + "default_", StringComparison.Ordinal) || key.StartsWith(Prefix + "scheduler.", StringComparison.Ordinal))
                    {
                        metadata[key] = value;
                    }
                }
            }

            resources.AddRange(source.OwnedResources);
        }

        var merged = new ModelSource
        {
            Format = primary.Format,
            Architecture = WanAdapter.Architecture,
            Metadata = metadata,
            Weights = weights,
            OwnedResources = resources,
            Path = primary.Path,
            FormatData = primary.FormatData,
            CompilationCaches = primary.CompilationCaches,
        };
        return new WanAdapter().Build(new ModelSourceSet { Primary = merged }, options);
    }

    /// <summary>
    /// The low-noise expert of a two-expert model given as a second transformer-only file: its transformer weights become the
    /// <c>transformer_2.</c> ones, with Wan 2.2 A14B's switch at 0.875 of the timesteps and its guidance (4 / 3, 40 steps) unless the
    /// primary already names them.
    /// </summary>
    private static void AddSecondTransformer(ModelSource source, Dictionary<string, WeightInfo> weights, Dictionary<string, object> metadata)
    {
        const string first = "transformer.";
        const string second = "transformer_2.";
        foreach (var (name, weight) in source.Weights)
        {
            if (name.StartsWith(first, StringComparison.Ordinal))
            {
                string canonical = second + name[first.Length..];
                weights[canonical] = weight.As(canonical);
            }
        }

        metadata.TryAdd(Prefix + "boundary_ratio", 0.875);
        metadata[Prefix + "default_guidance"] = metadata.ContainsKey(Prefix + "default_guidance_2") ? metadata[Prefix + "default_guidance"] : 4.0;
        metadata.TryAdd(Prefix + "default_guidance_2", 3.0);
        metadata[Prefix + "default_steps"] = 40;
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
