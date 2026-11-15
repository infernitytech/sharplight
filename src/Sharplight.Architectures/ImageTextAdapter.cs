using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Architectures;

/// <summary>
/// Dual-encoder image + text models (canonical arch <c>image_text</c>: CLIP, SigLIP): an image tower (<see cref="ImageTowerGraph"/>, weights
/// <c>vision.*</c>) and a text tower (<see cref="TextTowerGraph"/>, weights <c>text.*</c>) that embed into one space. Described as an
/// <see cref="ModelKinds.ImageEmbedding"/> model with <see cref="ModelComponentRoles.VisionEncoder"/>, <see cref="ModelComponentRoles.TextEncoder"/>,
/// <see cref="ModelComponentRoles.Tokenizer"/> and <see cref="ModelComponentRoles.FeatureExtractor"/> components.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>image_text.*</c>): per tower <c>vision.</c> / <c>text.</c> + <c>layer_count</c>, <c>hidden_size</c>, <c>ffn_hidden_size</c>,
/// <c>attention.head_count</c>, <c>norm_eps</c>, <c>mlp.activation</c>; vision <c>patch_size</c>, <c>image_size</c>, <c>pooling</c> ("cls" or "attention_head");
/// text <c>context_length</c>, <c>causal</c>, <c>pool</c> ("eos" pools the row of the highest token id, as CLIP does; "last" the last row),
/// <c>pad_token_id</c>; <c>scoring</c> ("softmax" or "sigmoid") of zero-shot scores; <see cref="CanonicalKeys.ImageProcessor"/>. Tokenizer
/// metadata is stored as the tokenizer component's attachment.
/// </remarks>
public sealed class ImageTextAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "image_text";

    private const string TokenizerAttachment = "tokenizer/tokenizer.json";

    /// <inheritdoc/>
    public string Name => "Dual-encoder image and text model";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        TowerOptions Tower(string kind) => new(
            src.Arch<int>($"{kind}.layer_count"), src.Arch<int>($"{kind}.hidden_size"), src.Arch<int>($"{kind}.attention.head_count"),
            src.Arch($"{kind}.norm_eps", 1e-6f), src.Arch($"{kind}.mlp.activation", "gelu"), kind + ".");
        var vision = Tower("vision");
        var text = Tower("text");
        int patch = src.Arch<int>("vision.patch_size");
        int imageSize = src.Arch<int>("vision.image_size");
        int length = src.Arch<int>("text.context_length");
        bool causal = src.Arch("text.causal", false);
        string pool = src.Arch("text.pool", "last");
        int padId = src.Arch("text.pad_token_id", 0);
        var imageGraph = ImageTowerGraph.Build(WellKnown.ImageEncodeGraph, src, new ImageTowerOptions(vision, patch, imageSize, 0, src.Arch("vision.pooling", "cls")));
        var textGraph = TextTowerGraph.Build(src, text, length, causal);

        var tokenizer = TokenizerData.FromMetadata(src.Metadata);
        var attachments = new Dictionary<string, IWeightSource>(src.Attachments, StringComparer.Ordinal)
        {
            [TokenizerAttachment] = new MemoryWeightSource(tokenizer.ToJson()),
        };
        var metadata = src.Metadata.Where(e => !e.Key.StartsWith("tokenizer.", StringComparison.Ordinal)).ToDictionary(StringComparer.Ordinal);
        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var description = new ModelDescription
        {
            Kind = ModelKinds.ImageEmbedding,
            Config = ComponentConfig.Of(("scoring", src.Arch("scoring", "softmax"))).ToDictionary(),
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
                    Graphs = [WellKnown.ImageEncodeGraph],
                    WeightPrefixes = ["vision."],
                    Config = ComponentConfig.Of(("patchSize", patch), ("imageSize", imageSize)),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.TextEncoder,
                    Graphs = [WellKnown.TextEncodeGraph],
                    WeightPrefixes = ["text."],
                    Config = ComponentConfig.Of(("contextLength", length), ("pool", pool), ("padTokenId", padId)),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.Tokenizer] = TokenizerAttachment },
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
                [WellKnown.ImageEncodeGraph] = imageGraph,
                [WellKnown.TextEncodeGraph] = textGraph,
            },
            Attachments = attachments,
            SourceMetadata = metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
