using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Add-on networks of image generators that are loaded as auxiliary sources and kept as components of their own beside the model's: the
/// tiny autoencoder (role <see cref="AuxiliaryRoles.TinyVae"/>, architecture <see cref="TinyAutoencoderArchitecture"/>, weights <c>tiny_vae.*</c>,
/// metadata <c>tiny_autoencoder.*</c>) and the ControlNet (role <see cref="AuxiliaryRoles.ControlNet"/>, architecture
/// <see cref="ControlNetArchitecture"/>, weights and metadata <c>controlnet.*</c>). Their weights and metadata are namespaced, so they are
/// merged into the primary source unchanged; the merged source is what the model is rebuilt from and saved as, so a package keeps them.
/// </summary>
internal static class DiffusionAddOns
{
    /// <summary>Canonical architecture of tiny autoencoder sources (TAESD).</summary>
    public const string TinyAutoencoderArchitecture = "tiny_autoencoder";

    /// <summary>Canonical architecture of ControlNet sources.</summary>
    public const string ControlNetArchitecture = "controlnet";

    private static readonly Dictionary<string, (string Architecture, string WeightPrefix, string MetadataPrefix)> s_roles = new(StringComparer.Ordinal)
    {
        [AuxiliaryRoles.TinyVae] = (TinyAutoencoderArchitecture, TinyAutoencoderHyperParameters.WeightPrefix, TinyAutoencoderHyperParameters.Prefix),
        [AuxiliaryRoles.ControlNet] = (ControlNetArchitecture, ControlNetHyperParameters.WeightPrefix, ControlNetHyperParameters.Prefix),
    };

    /// <summary>Whether an auxiliary role names an add-on.</summary>
    /// <param name="role">The auxiliary role.</param>
    public static bool IsAddOnRole(string role) => s_roles.ContainsKey(role);

    /// <summary>The primary source with the set's add-on sources merged in (the primary itself when there is none).</summary>
    /// <param name="sources">The sources.</param>
    /// <param name="acceptsControlNet">Whether the model takes a ControlNet (the UNet models).</param>
    /// <exception cref="NotSupportedException">An add-on role holds a source of another kind, or a ControlNet for a model without one.</exception>
    public static ModelSource Merge(ModelSourceSet sources, bool acceptsControlNet)
    {
        ArgumentNullException.ThrowIfNull(sources);
        return Merge(sources.Primary, sources.Auxiliary, acceptsControlNet);
    }

    /// <summary>A source with the add-on sources among <paramref name="auxiliary"/> merged in.</summary>
    /// <param name="primary">The (merged) primary source.</param>
    /// <param name="auxiliary">Auxiliary sources by role; roles that are not add-ons are ignored.</param>
    /// <param name="acceptsControlNet">Whether the model takes a ControlNet (the UNet models).</param>
    /// <exception cref="NotSupportedException">An add-on role holds a source of another kind, or a ControlNet for a model without one.</exception>
    public static ModelSource Merge(ModelSource primary, IReadOnlyDictionary<string, ModelSource> auxiliary, bool acceptsControlNet)
    {
        var addOns = auxiliary.Where(pair => IsAddOnRole(pair.Key)).ToList();
        if (addOns.Count == 0)
        {
            return primary;
        }

        var metadata = new Dictionary<string, object>(primary.Metadata, StringComparer.Ordinal);
        var weights = new Dictionary<string, WeightInfo>(primary.Weights, StringComparer.Ordinal);
        var resources = new List<IDisposable>(primary.OwnedResources);
        foreach (var (role, source) in addOns)
        {
            var (architecture, weightPrefix, metadataPrefix) = s_roles[role];
            if (source.Architecture != architecture)
            {
                throw new NotSupportedException($"Auxiliary role '{role}': '{source.Path}' is a {source.Architecture} model, not a {architecture}.");
            }

            if (role == AuxiliaryRoles.ControlNet && !acceptsControlNet)
            {
                throw new NotSupportedException($"Not supported yet: ControlNets for {primary.Get(CanonicalKeys.Name, primary.Architecture)} ({primary.Architecture}); "
                    + "ControlNets condition the SD 1.x / 2.x / XL UNets.");
            }

            foreach (string name in weights.Keys.Where(n => n.StartsWith(weightPrefix, StringComparison.Ordinal)).ToList())
            {
                weights.Remove(name);
            }

            foreach (string key in metadata.Keys.Where(k => k.StartsWith(metadataPrefix, StringComparison.Ordinal)).ToList())
            {
                metadata.Remove(key);
            }

            foreach (var (name, weight) in source.Weights)
            {
                weights[name] = weight;
            }

            foreach (var (key, value) in source.Metadata.Where(pair => pair.Key.StartsWith(metadataPrefix, StringComparison.Ordinal)))
            {
                metadata[key] = value;
            }

            resources.AddRange(source.OwnedResources);
        }

        return new ModelSource
        {
            Format = primary.Format,
            Architecture = primary.Architecture,
            Metadata = metadata,
            Weights = weights,
            Attachments = primary.Attachments,
            OwnedResources = resources,
            Path = primary.Path,
            FormatData = primary.FormatData,
            CompilationCaches = primary.CompilationCaches,
        };
    }

    /// <summary>Adds the tiny autoencoder's graphs when the source has one (the ControlNet's graph is the UNet adapter's).</summary>
    /// <param name="src">The merged source.</param>
    /// <param name="graphs">The model's graphs.</param>
    /// <param name="latentChannels">The model's latent channels (of its VAE graphs).</param>
    /// <param name="downscale">The model's pixels per latent.</param>
    /// <param name="latentHeight">Latent height of the built size.</param>
    /// <param name="latentWidth">Latent width of the built size.</param>
    /// <exception cref="NotSupportedException">The tiny autoencoder is for another latent space.</exception>
    public static void AddTinyAutoencoder(ModelSource src, IDictionary<string, ComputeGraph> graphs, int latentChannels, int downscale, long latentHeight, long latentWidth)
    {
        if (!TinyAutoencoderHyperParameters.IsPresent(src))
        {
            return;
        }

        var hp = TinyAutoencoderHyperParameters.Read(src);
        if (hp.LatentChannels != latentChannels || hp.Downscale != downscale)
        {
            throw new NotSupportedException($"The tiny autoencoder decodes {hp.LatentChannels}-channel latents at 1/{hp.Downscale} of the image size; "
                + $"{src.Get(CanonicalKeys.Name, src.Architecture)} has {latentChannels}-channel latents at 1/{downscale} (TAESD / TAESDXL are for SD 1.x / 2.x / XL, "
                + "TAESD3 for SD3, TAEF1 for FLUX.1).");
        }

        if (hp.HasDecoder)
        {
            graphs[DiffusionGraphs.TinyVaeDecode] = TinyAutoencoderGraphBuilder.BuildDecoder(src.Weight, hp, latentHeight, latentWidth);
        }

        if (hp.HasEncoder)
        {
            graphs[DiffusionGraphs.TinyVaeEncode] = TinyAutoencoderGraphBuilder.BuildEncoder(src.Weight, hp, latentHeight * downscale, latentWidth * downscale);
        }
    }

    /// <summary>The description with the components of the add-on graphs among <paramref name="graphs"/>.</summary>
    /// <param name="description">The model's description.</param>
    /// <param name="graphs">The model's graphs.</param>
    public static ModelDescription WithAddOns(ModelDescription description, IReadOnlyDictionary<string, ComputeGraph> graphs)
    {
        var extra = Components(graphs, description.Components.FirstOrDefault(c => c.Role == ModelComponentRoles.Denoiser)).ToList();
        return extra.Count == 0 ? description : new ModelDescription
        {
            SchemaVersion = description.SchemaVersion,
            Kind = description.Kind,
            Components = [.. description.Components, .. extra],
            Config = description.Config,
        };
    }

    private static IEnumerable<ModelComponent> Components(IReadOnlyDictionary<string, ComputeGraph> graphs, ModelComponent? denoiser)
    {
        if (graphs.ContainsKey(DiffusionGraphs.DenoiseControl))
        {
            // The UNet's weights as well as the ControlNet's; as many rows per run as the plain denoiser.
            yield return new ModelComponent
            {
                Role = DiffusionGraphs.ControlNetRole,
                Graphs = [DiffusionGraphs.DenoiseControl],
                WeightPrefixes = [ControlNetHyperParameters.WeightPrefix],
                SymbolBounds = denoiser?.SymbolBounds ?? new Dictionary<string, long> { [DiffusionGraphs.BatchSymbol] = 2 },
            };
        }

        string[] tiny = [.. new[] { DiffusionGraphs.TinyVaeDecode, DiffusionGraphs.TinyVaeEncode }.Where(graphs.ContainsKey)];
        if (tiny.Length > 0)
        {
            yield return new ModelComponent
            {
                Role = DiffusionGraphs.TinyVaeRole,
                Graphs = tiny,
                WeightPrefixes = [TinyAutoencoderHyperParameters.WeightPrefix],
                SymbolBounds = new Dictionary<string, long> { [DiffusionGraphs.BatchSymbol] = 1 },
            };
        }
    }
}
