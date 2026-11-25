using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a Stable Audio Open model (canonical architecture <see cref="Architecture"/>, see <c>Sharplight.Import.Diffusers</c>).</summary>
/// <param name="Hidden">Width of the diffusion transformer.</param>
/// <param name="Layers">Its layers.</param>
/// <param name="Heads">Its attention heads.</param>
/// <param name="KvHeads">Key-value heads of its cross-attention.</param>
/// <param name="HeadDim">Width of a head.</param>
/// <param name="ConditionWidth">Width of the conditioning rows (the text encoder and the timing conditioner).</param>
/// <param name="LatentChannels">Values of a latent frame.</param>
/// <param name="Frames">The latent frames the transformer generates.</param>
/// <param name="NormEps">Layer norm epsilon of the transformer.</param>
/// <param name="RopeDims">The rotated head dimensions (partial rotary embedding).</param>
/// <param name="RopeBase">Rotary base.</param>
/// <param name="MinSeconds">The smallest number of seconds the timing conditioner encodes.</param>
/// <param name="MaxSeconds">The largest.</param>
/// <param name="SampleRate">Sample rate of the decoded audio.</param>
/// <param name="AudioChannels">Channels of the decoded audio.</param>
/// <param name="SamplesPerFrame">Audio samples per latent frame.</param>
/// <param name="VaeUpsampling">The decoder's upsampling strides, deep to shallow.</param>
/// <param name="SigmaMin">The smallest sigma of the sampler's schedule.</param>
/// <param name="SigmaMax">The largest.</param>
/// <param name="SigmaData">The sigma of the data.</param>
/// <param name="DefaultSteps">Default number of sampler steps.</param>
/// <param name="DefaultGuidance">Default classifier-free guidance scale.</param>
internal sealed record StableAudioParameters(int Hidden, int Layers, int Heads, int KvHeads, int HeadDim, int ConditionWidth, int LatentChannels, int Frames, float NormEps,
    int RopeDims, float RopeBase, float MinSeconds, float MaxSeconds, int SampleRate, int AudioChannels, int SamplesPerFrame, long[] VaeUpsampling, float SigmaMin,
    float SigmaMax, float SigmaData, int DefaultSteps, float DefaultGuidance)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "stable_audio";

    /// <summary>The key prefix of the hyper-parameters.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static StableAudioParameters From(ModelSource source) => new(
        source.Arch<int>("hidden_size"), source.Arch<int>("layer_count"), source.Arch<int>("head_count"), source.Arch<int>("kv_head_count"), source.Arch<int>("head_dim"),
        source.Arch<int>("condition_width"), source.Arch<int>("latent_channels"), source.Arch<int>("frames"), source.Arch("norm_eps", 1e-5f),
        source.Arch<int>("rope_dims"), source.Arch("rope_base", 10000f), source.Arch("timing.min_seconds", 0f), source.Arch<float>("timing.max_seconds"),
        source.Arch<int>("sample_rate"), source.Arch<int>("audio_channels"), source.Arch<int>("samples_per_frame"), source.Arch<long[]>("vae.upsampling_ratios"),
        source.Arch<float>("sigma_min"), source.Arch<float>("sigma_max"), source.Arch("sigma_data", 1f), source.Arch("default_steps", 100), source.Arch("default_guidance", 7f));
}
