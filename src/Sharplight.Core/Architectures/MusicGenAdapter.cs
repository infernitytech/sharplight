using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Architectures;

/// <summary>
/// MusicGen text-to-music models (canonical arch <c>musicgen</c>, kind <see cref="ModelKinds.MusicGeneration"/>): a T5 text encoder, an autoregressive transformer over
/// the codes of four EnCodec codebooks (delay pattern, classifier-free guidance) and the 32 kHz EnCodec decoder. Builds the graphs of <see cref="MusicGenerationGraphs"/>
/// (see <see cref="MusicGenGraph"/>; the codec decoder is <see cref="EncodecGraph.BuildDecoder"/> with non-causal convolutions) and describes the tokenizer.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>musicgen.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>attention.head_count</c>, <c>codebook_count</c>, <c>vocabulary</c>, <c>max_steps</c>, <c>norm_eps</c>,
/// <c>guidance_scale</c>, <c>frame_rate</c>; <c>text.{layer_count, hidden_size, attention.head_count, head_dim, norm_eps, relative_buckets, relative_max_distance, context_length}</c>;
/// the codec as the EnCodec keys under <c>codec.</c> (see <c>EncodecAdapter</c>, plus <c>codec.causal</c>). Weights: the T5 encoder as <c>text.*</c> (token_embedding.weight,
/// relative_bias.weight, layers.{i}.attention.{q,k,v,o}.weight, attention_norm.weight, mlp_norm.weight, mlp.{up,down}.weight, output_norm.weight, projection.{weight,bias}), the
/// decoder as <c>decoder.*</c> (embedding.{q}.weight, position_table.weight, layers.{i}.{self_attention, cross_attention}.{q,k,v,o}.weight, self_norm, cross_norm, mlp_norm
/// {weight,bias}, mlp.{up,down}.weight, output_norm, lm_head.{q}.weight) and the codec decoder as <c>codec.decoder.*</c> and <c>codec.quantizer.*</c>.
/// </remarks>
public sealed class MusicGenAdapter : IModelAdapter
{
    private const string TokenizerAttachment = "tokenizer/tokenizer.json";

    /// <inheritdoc/>
    public string Name => "MusicGen music generator";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [MusicGenParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = MusicGenParameters.From(src);
        var codec = MusicGenParameters.Codec(src);
        var states = MusicGenGraph.States(p, options);
        Func<string, WeightInfo?> codecWeights = name => src.Weight("codec." + name);
        var tokenizer = TokenizerData.FromMetadata(src.Metadata);
        var attachments = new Dictionary<string, IWeightSource>(src.Attachments, StringComparer.Ordinal)
        {
            [TokenizerAttachment] = new MemoryWeightSource(tokenizer.ToJson()),
        };
        var metadata = src.Metadata.Where(e => !e.Key.StartsWith("tokenizer.", StringComparison.Ordinal)).ToDictionary(StringComparer.Ordinal);
        int frames = 1500;
        var description = new ModelDescription
        {
            Kind = ModelKinds.MusicGeneration,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.TextEncoder,
                    Graphs = [MusicGenerationGraphs.TextGraph],
                    WeightPrefixes = ["text."],
                    Config = ComponentConfig.Of((MusicGenerationGraphs.MaxTokensKey, p.TextMaxTokens), (MusicGenerationGraphs.ConditionWidthKey, p.Hidden)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = p.TextMaxTokens },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.Tokenizer] = TokenizerAttachment },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Decoder,
                    Graphs = [MusicGenerationGraphs.StepGraph, MusicGenerationGraphs.UnconditionalStepGraph],
                    WeightPrefixes = ["decoder."],
                    Config = ComponentConfig.Of((MusicGenerationGraphs.CodebooksKey, p.Codebooks), (MusicGenerationGraphs.VocabularyKey, p.Vocabulary),
                        (MusicGenerationGraphs.MaxStepsKey, p.MaxSteps), (MusicGenerationGraphs.GuidanceKey, (double)p.Guidance),
                        (MusicGenerationGraphs.SampleRateKey, codec.SampleRate), (MusicGenerationGraphs.FrameRateKey, p.FrameRate)),
                    SymbolBounds = new Dictionary<string, long>
                    {
                        [WellKnown.TokenSymbol] = 1,
                        [WellKnown.ContextSymbol] = p.MaxSteps,
                        [WellKnown.EncoderSymbol] = p.TextMaxTokens,
                    },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vocoder,
                    Graphs = [AudioCodecGraphs.DecodeGraph],
                    WeightPrefixes = ["codec."],
                    Config = ComponentConfig.Of((AudioCodecGraphs.SampleRateKey, codec.SampleRate), (AudioCodecGraphs.SamplesPerFrameKey, (int)codec.SamplesPerFrame),
                        (AudioCodecGraphs.CodebooksKey, codec.Codebooks), (AudioCodecGraphs.DefaultCodebooksKey, codec.Codebooks), (AudioCodecGraphs.CodebookSizeKey, codec.CodebookSize)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = frames },
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
                [MusicGenerationGraphs.TextGraph] = MusicGenGraph.BuildText(p, src.Weight),
                [MusicGenerationGraphs.StepGraph] = MusicGenGraph.BuildStep(p, src.Weight, states, conditioned: true),
                [MusicGenerationGraphs.UnconditionalStepGraph] = MusicGenGraph.BuildStep(p, src.Weight, states, conditioned: false),
                [AudioCodecGraphs.DecodeGraph] = EncodecGraph.BuildDecoder(codec, codecWeights),
            },
            States = [.. states],
            Attachments = attachments,
            SourceMetadata = metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
