using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Promptable image segmenters of the SAM 2 family (canonical arch <c>sam2</c>): a Hiera backbone (windowed attention, max-pooled queries at the
/// stage changes, sparse global blocks, a windowed position embedding), a feature-pyramid neck, a prompt encoder (Fourier point and box embeddings) and
/// a two-way transformer mask decoder with high-resolution skip features. Builds <see cref="WellKnown.SegmentEncodeGraph"/> (the image stage) and
/// <see cref="WellKnown.SegmentDecodeGraph"/> (prompts to masks) for a fixed square input and describes the model as
/// <see cref="ModelKinds.ImageSegmentation"/>. The video memory modules of the checkpoints are not used.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>sam2.*</c>): <c>image_size</c>, <c>patch.{kernel,stride,padding}</c>, <c>position_background</c>, the per-stage lists
/// <c>embed_dims</c>, <c>blocks</c>, <c>heads</c>, <c>windows</c> ("96,192,384,768"), <c>global_blocks</c>, <c>query_stride</c>, <c>query_pool_stages</c>, <c>mlp_ratio</c>,
/// <c>norm_eps</c>, <c>neck.size</c>, <c>neck.top_down_levels</c>, <c>decoder.{head_count,layer_count,downsample,mask_tokens}</c>, <c>prompt.patch_size</c>.
/// Weights keep the names of the Hugging Face layout (<c>vision_encoder.backbone.*</c>, <c>vision_encoder.neck.convs.*</c>, <c>mask_decoder.*</c>,
/// <c>prompt_encoder.*</c>, <c>shared_image_embedding.positional_embedding</c>, <c>no_memory_embedding</c>).
/// </remarks>
public sealed class Sam2Adapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "sam2";

    /// <summary>The most prompt tokens (points, or box corners with a padding entry) one prompt can have.</summary>
    public const int MaxPromptTokens = 64;

    /// <inheritdoc/>
    public string Name => "Promptable segmenter (SAM 2)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = Sam2Parameters.From(src);
        var encode = Sam2ImageGraph.Build(src, p);
        var decode = Sam2DecoderGraph.Build(src, p);
        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        int maskSize = (int)decode.Outputs[WellKnown.SegmentMasksOutput].Shape[1].Fixed;
        var description = new ModelDescription
        {
            Kind = ModelKinds.ImageSegmentation,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = new Dictionary<string, JsonElement> { [ImageProcessor.ComponentKey] = JsonSerializer.SerializeToElement(processor) },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.VisionEncoder,
                    Graphs = [WellKnown.SegmentEncodeGraph],
                    Config = ComponentConfig.Of(("imageSize", p.ImageSize)),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.MaskDecoder,
                    Graphs = [WellKnown.SegmentDecodeGraph],
                    SymbolBounds = new Dictionary<string, long>(StringComparer.Ordinal) { [WellKnown.SegmentPointSymbol] = MaxPromptTokens },
                    Config = ComponentConfig.Of(("imageSize", p.ImageSize), ("maskSize", maskSize), ("maskCount", p.MaskTokens), ("maxPromptTokens", MaxPromptTokens)),
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph>
            {
                [WellKnown.SegmentEncodeGraph] = encode,
                [WellKnown.SegmentDecodeGraph] = decode,
            },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
