using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// VoxCPM2 text to speech (canonical architecture <see cref="Architecture"/>; OpenBMB's tokenizer-free diffusion autoregressive model): a text-semantic language model
/// (MiniCPM-4) reads the text tokens and the AudioVAE latents of reference or prompt audio (as the local encoder's rows), a residual acoustic language model reads the
/// fusion of its scalar-quantized output with the latents, and for every 4-frame patch of 64-wide latents (6.25 patches per second) the local diffusion transformer
/// solves a flow-matching ODE (Euler, classifier-free guidance with the zero-star rescaling) conditioned on both models' states and on the previous patch; a stop head
/// on the text-semantic state ends the speech. The AudioVAE V2 decodes the latents to 48 kHz audio (its encoder takes 16 kHz).
/// The language models are built by <see cref="DecoderAdapter"/> over embedding rows (the text-semantic model also embeds text tokens itself; the residual model's
/// graph and caches live under <see cref="ResidualPrefix"/>); <see cref="VoxCpm2Graphs"/> and <see cref="AudioVaeV2Graph"/> build the rest; <see cref="VoxCpm2Synthesizer"/> runs it.
/// </summary>
/// <remarks>
/// Hyper-parameters: the text-semantic model's <c>decoder.*</c> keys, the residual model's under <c>voxcpm2.residual.decoder.*</c>; <c>voxcpm2.{patch_size,feat_dim,max_length,
/// norm_eps}</c>, <c>voxcpm2.rope.base</c>, <c>voxcpm2.fsq.{latent,scale}</c>, the local transformers' <c>voxcpm2.{encoder,dit}.{hidden_size,ffn_hidden_size,head_count,
/// kv_head_count,head_dim,layer_count}</c>, <c>voxcpm2.dit.mean_mode</c>, the special tokens <c>voxcpm2.token.{audio_start,audio_end,reference_start,reference_end}</c>, the
/// defaults <c>voxcpm2.sampling.{guidance,steps}</c> and the AudioVAE's <c>voxcpm2.vae.{encoder_dim,encoder_rates,latent,decoder_dim,decoder_rates,sample_rate,
/// out_sample_rate,sample_rate_bucket}</c>. Weights: the text-semantic model's canonical decoder names (with <c>rope.freq_factors</c>, which the local transformers share),
/// <c>residual.</c> plus the residual model's, <c>feat_encoder.</c>, <c>feat_decoder.estimator.</c>, <c>fsq_layer.</c>, <c>{enc_to_lm,lm_to_dit,res_to_dit,fusion_concat}_proj.</c>,
/// <c>stop_proj.</c>, <c>stop_head.</c> and <c>audio_vae.</c> (see <see cref="AudioVaeV2Graph"/>).
/// </remarks>
internal sealed class VoxCpm2Adapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "voxcpm2";

    /// <summary>The model family recorded in the text model component's configuration.</summary>
    public const string Family = "voxcpm2";

    /// <summary>The role of the text-semantic language model.</summary>
    public const string TextModelRole = "text-model";

    /// <summary>The role of the residual acoustic language model.</summary>
    public const string ResidualRole = "residual-model";

    /// <summary>The role of the local feature encoder.</summary>
    public const string EncoderRole = "local-encoder";

    /// <summary>The role of the local diffusion transformer.</summary>
    public const string DitRole = "diffusion";

    /// <summary>The role of the projections between the models (graphs <see cref="VoxCpm2Graphs.FuseGraph"/> and <see cref="VoxCpm2Graphs.ConditionGraph"/>).</summary>
    public const string ProjectionRole = "projections";

    /// <summary>The role of the AudioVAE's decoder.</summary>
    public const string VaeDecoderRole = "vae-decoder";

    /// <summary>The role of the AudioVAE's encoder.</summary>
    public const string VaeEncoderRole = "vae-encoder";

    /// <summary>The prefix of the residual model's weights and caches.</summary>
    public const string ResidualPrefix = "residual.";

    /// <summary>The residual model's graph.</summary>
    public const string ResidualGraph = ResidualPrefix + WellKnown.DecodeMultimodalGraph;

    /// <summary>The AudioVAE decoder graph (one patch per run).</summary>
    public const string VaeDecodeGraph = "voxcpm2.vae.decode";

    /// <summary>The AudioVAE encoder graph (<see cref="EncodeLatentFrames"/> latent frames per run).</summary>
    public const string VaeEncodeGraph = "voxcpm2.vae.encode";

    /// <summary>The latent frames the AudioVAE encoder graph produces per run.</summary>
    public const int EncodeLatentFrames = 32;

    /// <summary>The most patches the local encoder takes per run (longer prompts run in several).</summary>
    public const int MaxEncodePatches = 256;

    /// <summary>The most rows the projection graph takes per run.</summary>
    public const int MaxFuseRows = 512;

    /// <inheritdoc/>
    public string Name => "VoxCPM2";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        const string A = Architecture + ".";
        var backboneSource = new ModelSource
        {
            Format = src.Format,
            Architecture = "decoder",
            Metadata = new Dictionary<string, object>(src.Metadata, StringComparer.Ordinal) { ["decoder.embedding_input"] = true },
            Weights = src.Weights,
            Attachments = src.Attachments,
            Path = src.Path,
        };
        var backbone = new DecoderAdapter().Build(new ModelSourceSet { Primary = backboneSource }, options);

        const string residualKeys = A + "residual.";
        var residualMetadata = src.Metadata.Where(kv => kv.Key.StartsWith(residualKeys, StringComparison.Ordinal))
            .ToDictionary(kv => kv.Key[residualKeys.Length..], kv => kv.Value, StringComparer.Ordinal);
        residualMetadata[CanonicalKeys.Architecture] = "decoder";
        residualMetadata["decoder.embedding_input"] = true;
        var residualSource = new ModelSource
        {
            Format = src.Format,
            Architecture = "decoder",
            Metadata = residualMetadata,
            Weights = src.Weights.Where(kv => kv.Key.StartsWith(ResidualPrefix, StringComparison.Ordinal))
                .ToDictionary(kv => kv.Key[ResidualPrefix.Length..], kv => kv.Value, StringComparer.Ordinal),
            Path = src.Path,
        };
        var residualParameters = DecoderHyperParameters.Read(residualSource);
        var residualBuilder = new DecoderGraphBuilder(residualSource, options, residualParameters, statePrefix: ResidualPrefix);

        var p = VoxCpm2Parameters.From(src);
        var weights = new Dictionary<string, WeightInfo>(backbone.Weights, StringComparer.Ordinal);
        WeightInfo? Weight(string name) => weights.GetValueOrDefault(name);
        var graphs = new Dictionary<string, ComputeGraph>(backbone.Graphs, StringComparer.Ordinal)
        {
            [ResidualGraph] = residualBuilder.BuildMultimodal(0),
            [VoxCpm2Graphs.EncodeGraph] = VoxCpm2Graphs.BuildEncoder(p, Weight),
            [VoxCpm2Graphs.DitGraph] = VoxCpm2Graphs.BuildDit(p, Weight),
            [VoxCpm2Graphs.FuseGraph] = VoxCpm2Graphs.BuildFuse(p, Weight),
            [VoxCpm2Graphs.ConditionGraph] = VoxCpm2Graphs.BuildCondition(p, Weight),
            [VaeDecodeGraph] = AudioVaeV2Graph.BuildDecoder(VaeDecodeGraph, p.Vae, Weight, p.PatchSize),
            [VaeEncodeGraph] = AudioVaeV2Graph.BuildEncoder(VaeEncodeGraph, p.Vae, Weight, (int)(EncodeLatentFrames * p.Vae.EncoderHop)),
        };

        var config = new List<(string, object)>
        {
            ("family", Family), ("hidden", p.Hidden), ("patchSize", p.PatchSize), ("featureSize", p.FeatureSize), ("ditWidth", p.Dit.Hidden),
            ("audioStart", src.Get<int>(A + "token.audio_start")), ("audioEnd", src.Get<int>(A + "token.audio_end")),
            ("referenceStart", src.Get<int>(A + "token.reference_start")), ("referenceEnd", src.Get<int>(A + "token.reference_end")),
            ("maxLength", src.Get<int>(A + "max_length")), ("meanMode", src.Get(A + "dit.mean_mode", false)),
            ("sampling.guidance", src.Get(A + "sampling.guidance", 2f)), ("sampling.steps", src.Get(A + "sampling.steps", 10)),
            ("inputSampleRate", p.Vae.SampleRate), ("encoderHop", (int)p.Vae.EncoderHop), ("decoderHop", (int)p.Vae.DecoderHop),
        };
        string[] ownPrefixes = [ResidualPrefix, "feat_encoder.", "feat_decoder.", "fsq_layer.", "enc_to_lm_proj.", "lm_to_dit_proj.", "res_to_dit_proj.", "fusion_concat_proj.",
            "stop_proj.", "stop_head.", AudioVaeV2Graph.WeightPrefix];
        var backbonePrefixes = backbone.Weights.Keys.Where(k => !ownPrefixes.Any(prefix => k.StartsWith(prefix, StringComparison.Ordinal)))
            .Select(k => k[..(k.IndexOf('.', StringComparison.Ordinal) + 1)]).Where(prefix => prefix.Length > 0).Distinct().Order(StringComparer.Ordinal).ToList();
        int encodeRows = MaxEncodePatches * (p.PatchSize + 1);
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechSynthesis,
            Components =
            [
                new ModelComponent
                {
                    Role = TextModelRole,
                    Graphs = [.. backbone.Graphs.Keys.Order(StringComparer.Ordinal)],
                    WeightPrefixes = backbonePrefixes,
                    Config = ComponentConfig.Of([.. config]),
                },
                new ModelComponent { Role = ModelComponentRoles.Tokenizer, MetadataPrefix = null },
                new ModelComponent { Role = ResidualRole, Graphs = [ResidualGraph], WeightPrefixes = [ResidualPrefix] },
                new ModelComponent
                {
                    Role = EncoderRole,
                    Graphs = [VoxCpm2Graphs.EncodeGraph],
                    WeightPrefixes = ["feat_encoder.", "enc_to_lm_proj."],
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = encodeRows, [VoxCpm2Graphs.PatchSymbol] = MaxEncodePatches },
                },
                new ModelComponent { Role = DitRole, Graphs = [VoxCpm2Graphs.DitGraph], WeightPrefixes = ["feat_decoder."] },
                new ModelComponent
                {
                    Role = ProjectionRole,
                    Graphs = [VoxCpm2Graphs.FuseGraph, VoxCpm2Graphs.ConditionGraph],
                    WeightPrefixes = ["fsq_layer.", "fusion_concat_proj.", "lm_to_dit_proj.", "res_to_dit_proj.", "stop_proj.", "stop_head."],
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFuseRows },
                },
                new ModelComponent { Role = VaeDecoderRole, Graphs = [VaeDecodeGraph], WeightPrefixes = [AudioVaeV2Graph.WeightPrefix + "decoder."] },
                new ModelComponent { Role = VaeEncoderRole, Graphs = [VaeEncodeGraph], WeightPrefixes = [AudioVaeV2Graph.WeightPrefix + "encoder."] },
            ],
            Config = ComponentConfig.Of(("sampleRate", p.Vae.OutSampleRate)),
        };
        return new ModelDefinition
        {
            Architecture = Architecture,
            Name = backbone.Name,
            Weights = weights,
            Graphs = graphs,
            States = [.. backbone.States, .. residualBuilder.States],
            Properties = backbone.Properties,
            SourceMetadata = backbone.SourceMetadata,
            Description = description,
            Attachments = backbone.Attachments,
            OwnedResources = backbone.OwnedResources,
        };
    }
}
