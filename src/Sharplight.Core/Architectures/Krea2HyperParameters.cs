using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Hyper-parameters of a Krea 2 single-stream DiT (grouped-query gated attention over [text, image], one shared time modulation plus a table
/// per block, SwiGLU, zero-centred RMS norms, 3-axis RoPE, a text-fusion stage over the tapped text-encoder layers), read from
/// <c>krea2.transformer.*</c>.
/// </summary>
internal sealed record Krea2HyperParameters
{
    /// <summary>Patchified latent channels (VAE channels · 2 · 2).</summary>
    public required int InChannels { get; init; }

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Key / value heads.</summary>
    public required int KeyValueHeads { get; init; }

    /// <summary>Blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Width of the text features (the text encoder's hidden size).</summary>
    public required int TextDim { get; init; }

    /// <summary>Attention heads of the text-fusion blocks.</summary>
    public required int TextHeads { get; init; }

    /// <summary>Key / value heads of the text-fusion blocks.</summary>
    public required int TextKeyValueHeads { get; init; }

    /// <summary>Text-fusion blocks across the tapped layers of every token.</summary>
    public required int LayerwiseBlocks { get; init; }

    /// <summary>Text-fusion blocks across the tokens.</summary>
    public required int RefinerBlocks { get; init; }

    /// <summary>Tapped text-encoder layers stacked per token.</summary>
    public required int TextLayers { get; init; }

    /// <summary>Rotated head dimensions per position axis (frame, row, column).</summary>
    public required int[] AxesDim { get; init; }

    /// <summary>RoPE base.</summary>
    public required float RopeTheta { get; init; }

    /// <summary>Norm epsilon.</summary>
    public required float NormEps { get; init; }

    /// <summary>Native image size in pixels.</summary>
    public required int SampleSize { get; init; }

    /// <summary>Latent patch side.</summary>
    public int PatchSize => 2;

    /// <summary>VAE latent channels.</summary>
    public int LatentChannels => InChannels / (PatchSize * PatchSize);

    /// <summary>Width per head.</summary>
    public int HeadDim => HiddenSize / HeadCount;

    /// <summary>Width per head of the text-fusion blocks.</summary>
    public int TextHeadDim => TextDim / TextHeads;

    /// <summary>Reads the keys.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>krea2.</c>).</param>
    public static Krea2HyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "transformer.";
        return new Krea2HyperParameters
        {
            InChannels = source.Get(p + "in_channels", 64),
            HiddenSize = source.Get<int>(p + "hidden_size"),
            HeadCount = source.Get<int>(p + "head_count"),
            KeyValueHeads = source.Get<int>(p + "key_value_heads"),
            Layers = source.Get<int>(p + "layers"),
            TextDim = source.Get<int>(p + "text_dim"),
            TextHeads = source.Get<int>(p + "text_heads"),
            TextKeyValueHeads = source.Get<int>(p + "text_key_value_heads"),
            LayerwiseBlocks = source.Get<int>(p + "layerwise_blocks"),
            RefinerBlocks = source.Get<int>(p + "refiner_blocks"),
            TextLayers = source.Get<int>(p + "text_layers"),
            AxesDim = source.Get<int[]>(p + "axes_dim", [32, 48, 48]),
            RopeTheta = source.Get(p + "rope_theta", 1000f),
            NormEps = source.Get(p + "norm_eps", 1e-5f),
            SampleSize = source.Get(p + "sample_size", 1024),
        };
    }
}
