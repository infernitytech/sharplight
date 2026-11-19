using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Architectures;

/// <summary>
/// ACE-Step 1.5 (canonical arch <c>acestep</c>, kind <see cref="ModelKinds.AudioGeneration"/>): a Qwen3 caption encoder, a condition encoder (lyrics, timbre, caption), a
/// diffusion transformer trained as a flow (the turbo checkpoint takes 8 Euler steps without guidance) over the 25 Hz latents of a 48 kHz stereo Oobleck autoencoder.
/// Builds the graphs of <see cref="AudioGenerationGraphs"/>. The 5 Hz language model planner and the audio tokenizer (cover, repaint and lego tasks) are not part of the model.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>acestep.*</c>): see <see cref="AceStepParameters"/> (the keys are the snake-case names of its members). Weights: <c>dit.*</c> (the diffusion
/// transformer, the leading axis of the scale-shift tables removed), <c>cond.*</c> (the condition encoder: <c>text_projector</c>, <c>lyric_encoder</c>, <c>timbre_encoder</c>),
/// <c>text.*</c> (the Qwen3 caption encoder: <c>embed_tokens</c>, <c>layers.{i}</c>, <c>norm</c>) and <c>vae.decoder.*</c> (weight normalization folded, snake parameters
/// flattened to <c>[C]</c>); the attachment <c>silence-latent</c> holds the silence latent (little-endian f32, <c>[frames, 64]</c>).
/// </remarks>
public sealed class AceStepAdapter : IModelAdapter
{
    /// <summary>The attachment that carries the tokenizer's JSON.</summary>
    public const string TokenizerAttachment = "tokenizer/json";

    private const int MaxTextTokens = 256;
    private const int MaxLyricTokens = 2048;
    private const int MaxFrames = 3000;
    private const int DecodeFrames = 256;
    private const int DecodeOverlap = 16;

    /// <inheritdoc/>
    public string Name => "ACE-Step audio generation";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [AceStepParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = AceStepParameters.From(src);
        var tokenizer = TokenizerData.FromMetadata(src.Metadata);
        var attachments = new Dictionary<string, IWeightSource>(src.Attachments, StringComparer.Ordinal)
        {
            [TokenizerAttachment] = new MemoryWeightSource(tokenizer.ToJson()),
        };
        var metadata = src.Metadata.Where(e => !e.Key.StartsWith("tokenizer.", StringComparison.Ordinal)).ToDictionary(StringComparer.Ordinal);
        int conditionRows = MaxLyricTokens + 1 + MaxTextTokens;
        var description = new ModelDescription
        {
            Kind = ModelKinds.AudioGeneration,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.TextEncoder,
                    Graphs = [AudioGenerationGraphs.TextGraph, AudioGenerationGraphs.LyricGraph],
                    WeightPrefixes = ["text."],
                    Config = ComponentConfig.Of((AudioGenerationGraphs.MaxTextTokensKey, MaxTextTokens), (AudioGenerationGraphs.MaxLyricTokensKey, MaxLyricTokens),
                        (AudioGenerationGraphs.InstructionKey, p.Instruction), (AudioGenerationGraphs.TextWidthKey, p.TextHidden)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxLyricTokens },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.Tokenizer] = TokenizerAttachment },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [AudioGenerationGraphs.ConditionGraph],
                    WeightPrefixes = ["cond.", "dit.condition_embedder."],
                    Config = ComponentConfig.Of((AudioGenerationGraphs.ReferenceFramesKey, p.ReferenceFrames), (AudioGenerationGraphs.ConditionWidthKey, p.Hidden)),
                    SymbolBounds = new Dictionary<string, long>
                    {
                        [AudioGenerationGraphs.LyricSymbol] = MaxLyricTokens,
                        [AudioGenerationGraphs.TextSymbol] = MaxTextTokens,
                    },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Denoiser,
                    Graphs = [AudioGenerationGraphs.DenoiseGraph],
                    WeightPrefixes = ["dit."],
                    Config = ComponentConfig.Of((AudioGenerationGraphs.SampleRateKey, p.SampleRate), (AudioGenerationGraphs.ChannelsKey, p.AudioChannels),
                        (AudioGenerationGraphs.SamplesPerFrameKey, p.SamplesPerFrame), (AudioGenerationGraphs.LatentChannelsKey, p.LatentChannels),
                        (AudioGenerationGraphs.ContextChannelsKey, p.InChannels - p.LatentChannels), (AudioGenerationGraphs.SilenceFramesKey, p.SilenceFrames),
                        (AudioGenerationGraphs.ChunkMaskValueKey, (double)p.ChunkMaskValue), (AudioGenerationGraphs.PatchSizeKey, p.PatchSize),
                        (AudioGenerationGraphs.MaxFramesKey, MaxFrames), (AudioGenerationGraphs.DefaultStepsKey, p.DefaultSteps),
                        (AudioGenerationGraphs.DefaultShiftKey, (double)p.DefaultShift), (AudioGenerationGraphs.ReferenceFramesKey, p.ReferenceFrames)),
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [AudioGenerationGraphs.SilenceAttachment] = AudioGenerationGraphs.SilenceAttachment },
                    SymbolBounds = new Dictionary<string, long>
                    {
                        [WellKnown.TokenSymbol] = MaxFrames,
                        [WellKnown.EncoderSymbol] = conditionRows,
                    },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vae,
                    Graphs = [AudioGenerationGraphs.DecodeGraph],
                    WeightPrefixes = ["vae."],
                    Config = ComponentConfig.Of((AudioGenerationGraphs.SampleRateKey, p.SampleRate), (AudioGenerationGraphs.ChannelsKey, p.AudioChannels),
                        (AudioGenerationGraphs.SamplesPerFrameKey, p.SamplesPerFrame), (AudioGenerationGraphs.DecodeFramesKey, DecodeFrames),
                        (AudioGenerationGraphs.DecodeOverlapKey, DecodeOverlap)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = DecodeFrames + (2 * DecodeOverlap) },
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
                [AudioGenerationGraphs.TextGraph] = AceStepGraph.BuildText(p, src.Weight),
                [AudioGenerationGraphs.LyricGraph] = AceStepGraph.BuildLyricEmbedding(src.Weight),
                [AudioGenerationGraphs.ConditionGraph] = AceStepGraph.BuildCondition(p, src.Weight),
                [AudioGenerationGraphs.DenoiseGraph] = AceStepGraph.BuildDenoiser(p, src.Weight),
                [AudioGenerationGraphs.DecodeGraph] = AceStepGraph.BuildDecoder(p, src.Weight),
            },
            Attachments = attachments,
            SourceMetadata = metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
