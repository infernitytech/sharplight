using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of an ACE-Step 1.5 model (canonical architecture <see cref="Architecture"/>, see <c>Sharplight.Import.AceStep</c>).</summary>
/// <param name="Hidden">Width of the diffusion transformer and the condition encoders.</param>
/// <param name="Intermediate">Width of their feed-forward layers.</param>
/// <param name="Layers">Layers of the diffusion transformer.</param>
/// <param name="LyricLayers">Layers of the lyric encoder.</param>
/// <param name="TimbreLayers">Layers of the timbre encoder.</param>
/// <param name="Heads">Attention heads.</param>
/// <param name="KvHeads">Key-value heads.</param>
/// <param name="HeadDim">Width of a head.</param>
/// <param name="Eps">Norm epsilon.</param>
/// <param name="RopeBase">Rotary base.</param>
/// <param name="SlidingWindow">The radius of the sliding-window attention layers (a row sees the rows at most this far away).</param>
/// <param name="Sliding">Per layer: true for sliding-window attention.</param>
/// <param name="PatchSize">Latent frames per transformer row.</param>
/// <param name="LatentChannels">Values of a latent frame.</param>
/// <param name="InChannels">Input channels of the patch embedding (the noisy latents, the source latents and the chunk mask).</param>
/// <param name="TextHidden">Width of the text encoder.</param>
/// <param name="TextLayers">Layers of the text encoder.</param>
/// <param name="TextIntermediate">Feed-forward width of the text encoder.</param>
/// <param name="TextHeads">Attention heads of the text encoder.</param>
/// <param name="TextKvHeads">Its key-value heads.</param>
/// <param name="TextHeadDim">Its head width.</param>
/// <param name="TextRopeBase">Its rotary base.</param>
/// <param name="TextEps">Its norm epsilon.</param>
/// <param name="SampleRate">Sample rate of the decoded audio.</param>
/// <param name="AudioChannels">Channels of the decoded audio.</param>
/// <param name="SamplesPerFrame">Audio samples per latent frame.</param>
/// <param name="VaeChannels">Base width of the autoencoder's decoder.</param>
/// <param name="VaeMultiples">Channel multiples of the autoencoder, shallow to deep.</param>
/// <param name="VaeUpsampling">The decoder's upsampling strides, deep to shallow.</param>
/// <param name="SilenceFrames">Frames of the silence latent.</param>
/// <param name="ReferenceFrames">Frames of the timbre reference of audio without reference.</param>
/// <param name="DefaultSteps">Default number of denoising steps.</param>
/// <param name="DefaultShift">Default timestep shift.</param>
/// <param name="ChunkMaskValue">The value of the chunk mask of a generation without source audio.</param>
/// <param name="Instruction">The instruction of the caption prompt.</param>
internal sealed record AceStepParameters(int Hidden, int Intermediate, int Layers, int LyricLayers, int TimbreLayers, int Heads, int KvHeads, int HeadDim, float Eps, float RopeBase,
    int SlidingWindow, bool[] Sliding, int PatchSize, int LatentChannels, int InChannels, int TextHidden, int TextLayers, int TextIntermediate, int TextHeads, int TextKvHeads,
    int TextHeadDim, float TextRopeBase, float TextEps, int SampleRate, int AudioChannels, int SamplesPerFrame, int VaeChannels, long[] VaeMultiples, long[] VaeUpsampling,
    int SilenceFrames, int ReferenceFrames, int DefaultSteps, float DefaultShift, float ChunkMaskValue, string Instruction)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "acestep";

    /// <summary>The attention window of sliding layers in the attention op's terms (rows less than this far away).</summary>
    public long AttentionWindow => SlidingWindow + 1;

    /// <summary>The latent frames per second.</summary>
    public double FrameRate => (double)SampleRate / SamplesPerFrame;

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static AceStepParameters From(ModelSource source) => new(
        source.Arch<int>("hidden_size"), source.Arch<int>("intermediate_size"), source.Arch<int>("layer_count"), source.Arch<int>("lyric_layer_count"),
        source.Arch<int>("timbre_layer_count"), source.Arch<int>("head_count"), source.Arch<int>("kv_head_count"), source.Arch<int>("head_dim"),
        source.Arch<float>("norm_eps"), source.Arch<float>("rope_base"), source.Arch<int>("sliding_window"),
        [.. source.Arch<long[]>("sliding_layers").Select(v => v != 0)], source.Arch<int>("patch_size"), source.Arch<int>("latent_channels"),
        source.Arch<int>("in_channels"), source.Arch<int>("text.hidden_size"), source.Arch<int>("text.layer_count"), source.Arch<int>("text.intermediate_size"),
        source.Arch<int>("text.head_count"), source.Arch<int>("text.kv_head_count"), source.Arch<int>("text.head_dim"), source.Arch<float>("text.rope_base"),
        source.Arch<float>("text.norm_eps"), source.Arch<int>("sample_rate"), source.Arch<int>("audio_channels"), source.Arch<int>("samples_per_frame"),
        source.Arch<int>("vae.channels"), source.Arch<long[]>("vae.channel_multiples"), source.Arch<long[]>("vae.upsampling_ratios"),
        source.Arch<int>("silence_frames"), source.Arch<int>("reference_frames"), source.Arch<int>("default_steps"), source.Arch<float>("default_shift"),
        source.Arch<float>("chunk_mask_value"), source.Arch("instruction", "Fill the audio semantic mask based on the given conditions:"));
}
