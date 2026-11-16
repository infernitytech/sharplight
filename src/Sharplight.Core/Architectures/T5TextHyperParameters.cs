using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Hyper-parameters of a T5 text encoder used as a diffusion text encoder (FLUX, SD3), read from the encoder-decoder keys of
/// <see cref="EncoderDecoderAdapter"/> under a component prefix (<c>{prefix}{component}.hidden_size</c>, ...) plus the prompt length.
/// </summary>
internal sealed record T5TextHyperParameters
{
    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Encoder layers.</summary>
    public required int LayerCount { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Width per head (T5's d_kv).</summary>
    public required int HeadDim { get; init; }

    /// <summary>RMS norm epsilon.</summary>
    public required float NormEps { get; init; }

    /// <summary>Feed-forward activation ("relu", "gelu", "gelu_tanh", "silu").</summary>
    public required string Activation { get; init; }

    /// <summary>Distance from which the relative position bias uses its last bucket.</summary>
    public required int MaxDistance { get; init; }

    /// <summary>Tokens per prompt (the padded length the graph is built for).</summary>
    public required int ContextLength { get; init; }

    /// <summary>Reads the keys of one component.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>flux.</c>).</param>
    /// <param name="component">The component name (<c>text_encoder_2</c>).</param>
    public static T5TextHyperParameters Read(ModelSource source, string prefix, string component)
    {
        string p = $"{prefix}{component}.";
        int hidden = source.Get<int>(p + "hidden_size");
        int heads = source.Get<int>(p + "attention.head_count");
        return new T5TextHyperParameters
        {
            HiddenSize = hidden,
            LayerCount = source.Get<int>(p + "encoder.layer_count"),
            HeadCount = heads,
            HeadDim = source.Get(p + "attention.head_dim", hidden / heads),
            NormEps = source.Get(p + "norm_eps", 1e-6f),
            Activation = source.Get(p + "mlp.activation", "gelu_tanh"),
            MaxDistance = source.Get(p + "relative_attention.max_distance", 128),
            ContextLength = source.Get(p + "prompt_length", 512),
        };
    }
}
