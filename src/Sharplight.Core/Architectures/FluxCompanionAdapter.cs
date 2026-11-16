using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Completes a FLUX transformer with companion files loaded as auxiliary sources (transformer-only GGUF / single files are
/// distributed without their text encoders and VAE): the CLIP-L text encoder (role <see cref="AuxiliaryRoles.TextEncoder"/>,
/// architecture <c>clip_text_encoder</c>), the T5 encoder (role <see cref="AuxiliaryRoles.SecondTextEncoder"/>, any
/// <c>encoder_decoder</c> source such as a T5 encoder GGUF) and the VAE (role <see cref="AuxiliaryRoles.Vae"/>, architecture
/// <c>kl_autoencoder</c>). Their canonical weights and metadata are renamed into the <c>flux</c> vocabulary of
/// <see cref="FluxAdapter"/> (a companion replaces what the primary source already has), and the merged source is built by it, so
/// the model is rebuilt for other sizes and saved as one package like a complete checkpoint.
/// </summary>
public sealed class FluxCompanionAdapter : ICompositeModelAdapter
{
    /// <summary>Canonical architecture of a stand-alone CLIP text encoder source (weights without prefix, metadata <c>clip_text_encoder.*</c>).</summary>
    public const string ClipTextArchitecture = "clip_text_encoder";

    /// <summary>Canonical architecture of a stand-alone KL autoencoder source (weights <c>vae.*</c>, metadata <c>kl_autoencoder.vae.*</c>).</summary>
    public const string AutoencoderArchitecture = "kl_autoencoder";

    private const string T5Architecture = "encoder_decoder";
    private const string Prefix = FluxAdapter.Architecture + ".";
    private const string TokenizerKeys = "tokenizer.";

    /// <summary>The VAE keys a family fixes (the latent scaling the transformer was trained with) and a stand-alone VAE file cannot know.</summary>
    private static readonly string[] s_familyVaeKeys = ["vae.scaling_factor", "vae.shift_factor", "vae.latent_rgb_factors"];

    /// <inheritdoc/>
    public string Name => "FLUX with companion files";

    /// <inheritdoc/>
    public bool CanLoad(ModelSourceSet sources)
    {
        ArgumentNullException.ThrowIfNull(sources);
        if (sources.Primary.Architecture != FluxAdapter.Architecture || sources.Auxiliary.Count == 0)
        {
            return false;
        }

        foreach (var (role, source) in sources.Auxiliary)
        {
            bool known = role switch
            {
                AuxiliaryRoles.TextEncoder => source.Architecture == ClipTextArchitecture,
                AuxiliaryRoles.SecondTextEncoder => source.Architecture == T5Architecture,
                AuxiliaryRoles.Vae => source.Architecture == AutoencoderArchitecture,
                _ => false,
            };
            if (!known)
            {
                return false;
            }
        }

        return true;
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
            switch (role)
            {
                case AuxiliaryRoles.TextEncoder:
                    Replace(weights, "text_encoder.");
                    AddWeights(weights, source, name => "text_encoder." + name);
                    CopyMetadata(source, metadata, ClipTextArchitecture + ".", Prefix + "text_encoder.");
                    CopyMetadata(source, metadata, TokenizerKeys, TokenizerKeys);
                    break;
                case AuxiliaryRoles.SecondTextEncoder:
                    Replace(weights, "text_encoder_2.");
                    AddWeights(weights, source, name => name.StartsWith("encoder.", StringComparison.Ordinal) || name == "token_embedding.weight"
                        ? "text_encoder_2." + name
                        : null);
                    int? promptLength = metadata.TryGetValue(Prefix + "text_encoder_2.prompt_length", out var length) ? Convert.ToInt32(length, System.Globalization.CultureInfo.InvariantCulture) : null;
                    CopyMetadata(source, metadata, T5Architecture + ".", Prefix + "text_encoder_2.");
                    if (promptLength is { } kept)
                    {
                        metadata[Prefix + "text_encoder_2.prompt_length"] = kept;
                    }

                    CopyMetadata(source, metadata, TokenizerKeys, Prefix + "tokenizer_2." + TokenizerKeys);
                    break;
                default:
                    Replace(weights, "vae.");
                    AddWeights(weights, source, name => name.StartsWith("vae.", StringComparison.Ordinal) ? name : null);
                    var family = s_familyVaeKeys.Where(k => metadata.ContainsKey(Prefix + k)).ToDictionary(k => k, k => metadata[Prefix + k]);
                    CopyMetadata(source, metadata, AutoencoderArchitecture + ".vae.", Prefix + "vae.");
                    foreach (var (key, value) in family)
                    {
                        metadata[Prefix + key] = value;
                    }

                    break;
            }

            resources.AddRange(source.OwnedResources);
        }

        var merged = new ModelSource
        {
            Format = primary.Format,
            Architecture = FluxAdapter.Architecture,
            Metadata = metadata,
            Weights = weights,
            OwnedResources = resources,
            Path = primary.Path,
            FormatData = primary.FormatData,
            CompilationCaches = primary.CompilationCaches,
        };
        return new FluxAdapter().Build(new ModelSourceSet { Primary = merged }, options);
    }

    private static void Replace(Dictionary<string, WeightInfo> weights, string prefix)
    {
        foreach (string name in weights.Keys.Where(n => n.StartsWith(prefix, StringComparison.Ordinal)).ToList())
        {
            weights.Remove(name);
        }
    }

    private static void AddWeights(Dictionary<string, WeightInfo> weights, ModelSource source, Func<string, string?> rename)
    {
        foreach (var (name, weight) in source.Weights)
        {
            if (rename(name) is { } canonical)
            {
                weights[canonical] = weight.As(canonical);
            }
        }
    }

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
