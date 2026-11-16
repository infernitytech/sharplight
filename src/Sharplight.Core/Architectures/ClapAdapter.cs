using System.Text.Json;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Architectures;

/// <summary>
/// CLAP audio-text embedding models (canonical arch <c>clap</c>, kind <see cref="ModelKinds.AudioEmbedding"/>): the HTS-AT Swin audio tower and a RoBERTa text tower
/// projected into one space. Builds <see cref="AudioEmbedderGraphs.AudioGraph"/> and <see cref="AudioEmbedderGraphs.TextGraph"/> (see <see cref="ClapGraph"/>) and
/// describes the log-mel feature extractor (filter bank attachment <see cref="MelFiltersAttachment"/>) and the tokenizer.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>clap.*</c>): <c>sample_rate</c>, <c>clip_samples</c>, <c>fft_size</c>, <c>hop_length</c>, <c>mel_bins</c>, <c>embedding_length</c>,
/// <c>audio.{spec_size, patch_size, window_size, embed_dim, depths, head_counts, mlp_ratio, norm_eps}</c>, <c>text.{layer_count, hidden_size, attention.head_count, norm_eps,
/// context_length, position_offset}</c>. Weights: <c>audio.batch_norm.{scale,shift}</c> (the mel batch normalization folded), <c>audio.patch_embed.{weight [dim, patch²],bias}</c>,
/// <c>audio.patch_norm</c>, per block <c>audio.layers.{s}.blocks.{b}.</c> with <c>norm1</c>, <c>{q,k,v,o}.{weight,bias}</c>, <c>relative_bias</c> [(2w-1)², heads], <c>norm2</c>,
/// <c>mlp.{up,down}</c>, per stage <c>audio.layers.{s}.merge_norm</c> and <c>merge.weight</c>, <c>audio.norm</c>, <c>audio.projection.{1,2}</c>; <c>text.token_embedding.weight</c>,
/// <c>text.position_embedding.weight</c>, <c>text.token_type_embedding.weight</c>, <c>text.embedding_norm</c>, per layer <c>text.layers.{i}.</c> with <c>attention.{q,k,v,o}</c>,
/// <c>attention_norm</c>, <c>mlp.{up,down}</c>, <c>mlp_norm</c>, then <c>text.pooler</c>, <c>text.projection.{1,2}</c>; <c>logit_scale_audio</c> and <c>logit_scale_text</c> (logarithms).
/// </remarks>
public sealed class ClapAdapter : IModelAdapter
{
    /// <summary>The attachment holding the mel filter bank.</summary>
    public const string MelFiltersAttachment = "feature-extractor/mel-filters";

    private const string TokenizerAttachment = "tokenizer/tokenizer.json";

    /// <inheritdoc/>
    public string Name => "CLAP audio-text model";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [ClapParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = ClapParameters.From(src);
        if (!src.Attachments.ContainsKey(MelFiltersAttachment))
        {
            throw new NotSupportedException($"{src.Architecture}: needs the '{MelFiltersAttachment}' attachment.");
        }

        var tokenizer = TokenizerData.FromMetadata(src.Metadata);
        var attachments = new Dictionary<string, IWeightSource>(src.Attachments, StringComparer.Ordinal)
        {
            [TokenizerAttachment] = new MemoryWeightSource(tokenizer.ToJson()),
        };
        var metadata = src.Metadata.Where(e => !e.Key.StartsWith("tokenizer.", StringComparison.Ordinal)).ToDictionary(StringComparer.Ordinal);
        var features = new Dictionary<string, JsonElement>(StringComparer.Ordinal)
        {
            ["type"] = Json("\"log-mel\""),
            ["sampleRate"] = Json(p.SampleRate),
            ["fftSize"] = Json(p.FftSize),
            ["hopLength"] = Json(p.HopLength),
            ["melBins"] = Json(p.MelBins),
            ["center"] = Json("true"),
            ["padMode"] = Json("\"reflect\""),
            ["logBase"] = Json("\"10\""),
            ["logFloor"] = Json("1e-10"),
            ["logScale"] = Json("10"),
        };
        var description = new ModelDescription
        {
            Kind = ModelKinds.AudioEmbedding,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = features,
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.MelFilters] = MelFiltersAttachment },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [AudioEmbedderGraphs.AudioGraph],
                    WeightPrefixes = ["audio.", "logit_scale_audio"],
                    Config = ComponentConfig.Of((AudioEmbedderGraphs.SampleRateKey, p.SampleRate), (AudioEmbedderGraphs.ClipSamplesKey, p.ClipSamples),
                        (AudioEmbedderGraphs.FramesKey, p.Frames), (AudioEmbedderGraphs.DimensionsKey, p.Dimensions)),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.TextEncoder,
                    Graphs = [AudioEmbedderGraphs.TextGraph],
                    WeightPrefixes = ["text.", "logit_scale_text"],
                    Config = ComponentConfig.Of((AudioEmbedderGraphs.PositionOffsetKey, p.PositionOffset), (AudioEmbedderGraphs.MaxTokensKey, p.TextMaxTokens)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = p.TextMaxTokens },
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
            Graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal)
            {
                [AudioEmbedderGraphs.AudioGraph] = ClapGraph.BuildAudio(p, src.Weight),
                [AudioEmbedderGraphs.TextGraph] = ClapGraph.BuildText(p, src.Weight),
            },
            Attachments = attachments,
            SourceMetadata = metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }

    private static JsonElement Json(object value) => JsonSerializer.SerializeToElement(JsonDocument.Parse(value is string text ? text : Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture)!).RootElement);
}
