using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>
/// Builds SeedVR2 (one-step video restoration and super-resolution, 3B and 7B; architecture <c>seedvr</c>) as a
/// <see cref="ModelKinds.VideoEnhancement"/> model: the restoration DiT (<see cref="SeedVrTransformerGraphBuilder"/>), the causal video
/// VAE's encoder and decoder (<see cref="SeedVrVaeGraphBuilder"/>) and the wavelet colour transfer (<see cref="WaveletColorGraphBuilder"/>),
/// as the graphs of <see cref="SeedVrGraphs"/> for one clip size; <see cref="SeedVrEnhancer"/> runs them. Other sizes are rebuilt through the
/// dimensions <see cref="VideoDiffusionGraphs.FramesDimension"/> (latent frames), <see cref="VideoDiffusionGraphs.Height"/> and
/// <see cref="VideoDiffusionGraphs.Width"/> (pixels, multiples of 16).
/// </summary>
/// <remarks>
/// Canonical vocabulary: the transformer as documented on <see cref="SeedVrTransformerGraphBuilder"/>, the VAE as on
/// <see cref="SeedVrVaeGraphBuilder"/>, the text embeddings <c>text_embedding.positive</c> (and <c>.negative</c>) [L, text width]. Metadata:
/// <c>seedvr.transformer.*</c> (<see cref="SeedVrHyperParameters"/>), <c>seedvr.vae.*</c> (<see cref="SeedVrVaeHyperParameters"/>),
/// <c>seedvr.timestep</c> (of the one-step sampler), <c>seedvr.color_correction</c> and <c>seedvr.default_scale</c>.
/// </remarks>
public sealed class SeedVrAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "seedvr";

    /// <summary>Default size of the graphs of a freshly loaded model (rebuilt per request size): 17 frames of 832 × 480.</summary>
    private const int DefaultLatentFrames = 5;
    private const int DefaultHeight = 480;
    private const int DefaultWidth = 832;

    private const string Prefix = Architecture + ".";
    private const string TransformerPrefix = "transformer.";
    private const string VaePrefix = "vae.";
    private const string TextPrefix = "text_embedding.";

    /// <inheritdoc/>
    public string Name => "SeedVR2 (one-step restoration DiT + causal video VAE)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures { get; } = [Architecture];

    /// <summary>Pixels per side the graphs need sizes to be multiples of: the VAE's compression times the patch size.</summary>
    /// <param name="vae">The VAE hyper-parameters.</param>
    /// <param name="transformer">The transformer hyper-parameters.</param>
    internal static int SizeMultiple(SeedVrVaeHyperParameters vae, SeedVrHyperParameters transformer) => vae.SpatialDownscale * transformer.PatchSize[1];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(options);
        var src = sources.Primary;
        var transformer = SeedVrHyperParameters.Read(src, Prefix);
        var vae = SeedVrVaeHyperParameters.Read(src, Prefix);
        int multiple = SizeMultiple(vae, transformer);
        int height = (int)options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.Height, DefaultHeight);
        int width = (int)options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.Width, DefaultWidth);
        int latentFrames = (int)options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.FramesDimension, DefaultLatentFrames);
        if (height <= 0 || width <= 0 || height % multiple != 0 || width % multiple != 0 || latentFrames <= 0)
        {
            throw new ArgumentException($"Video size {width}x{height} must be a positive multiple of {multiple} pixels for {src.Get(CanonicalKeys.Name, src.Architecture)}.");
        }

        int latentHeight = height / vae.SpatialDownscale;
        int latentWidth = width / vae.SpatialDownscale;
        int frames = (latentFrames - 1) * vae.TemporalDownscale + 1;
        var graphs = new Dictionary<string, ComputeGraph>
        {
            [SeedVrGraphs.Restore] = SeedVrTransformerGraphBuilder.Build(src.Weight, transformer, latentFrames, latentHeight, latentWidth),
            [SeedVrGraphs.VaeEncode] = SeedVrVaeGraphBuilder.BuildEncoder(src.Weight, vae, frames, height, width),
            [SeedVrGraphs.VaeDecode] = SeedVrVaeGraphBuilder.BuildDecoder(src.Weight, vae, latentFrames, latentHeight, latentWidth),
            [SeedVrGraphs.ColorCorrect] = WaveletColorGraphBuilder.Build(frames, height, width),
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = graphs,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = Describe(src, transformer, vae),
        };
    }

    private static ModelDescription Describe(ModelSource src, SeedVrHyperParameters transformer, SeedVrVaeHyperParameters vae)
    {
        var model = new ComponentConfigBuilder()
            .Add(SeedVrConfigKeys.Timestep, src.Get(Prefix + "timestep", 1000.0))
            .Add(SeedVrConfigKeys.ColorCorrection, src.Get(Prefix + "color_correction", "wavelet"))
            .Add(SeedVrConfigKeys.DefaultScale, src.Get(Prefix + "default_scale", 2.0))
            .Add(SeedVrConfigKeys.SizeMultiple, SizeMultiple(vae, transformer))
            .Add(SeedVrConfigKeys.FrameMultiple, vae.TemporalDownscale);
        return new ModelDescription
        {
            Kind = ModelKinds.VideoEnhancement,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Transformer,
                    Graphs = [SeedVrGraphs.Restore],
                    WeightPrefixes = [TransformerPrefix, TextPrefix],
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.LatentChannels, transformer.OutChannels).Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vae,
                    Graphs = [SeedVrGraphs.VaeEncode, SeedVrGraphs.VaeDecode, SeedVrGraphs.ColorCorrect],
                    WeightPrefixes = [VaePrefix],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.LatentChannels, vae.LatentChannels)
                        .Add(SeedVrConfigKeys.SpatialDownscale, vae.SpatialDownscale)
                        .Add(SeedVrConfigKeys.TemporalDownscale, vae.TemporalDownscale)
                        .Build(),
                },
            ],
            Config = model.Build(),
        };
    }
}
