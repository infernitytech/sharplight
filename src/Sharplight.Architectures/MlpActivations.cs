namespace Sharplight.Architectures;

/// <summary>Activation factory: maps the <c>mlp.activation</c> name of a model to its <see cref="IMlpActivation"/>.</summary>
internal static class MlpActivations
{
    /// <summary>
    /// Creates the activation. Names: <c>silu</c>, <c>gelu</c> (erf), <c>gelu_tanh</c>, <c>relu</c>, <c>relu2</c> (squared ReLU),
    /// <c>xielu</c>, <c>swiglu_oai</c>.
    /// </summary>
    /// <exception cref="NotSupportedException">The name is unknown.</exception>
    public static IMlpActivation Create(DecoderHyperParameters hp) => hp.Activation switch
    {
        "silu" => new UnaryMlpActivation("Silu"),
        "gelu" => new UnaryMlpActivation("Gelu"),
        "gelu_tanh" => new UnaryMlpActivation("GeluTanh"),
        "relu" => new UnaryMlpActivation("Relu"),
        "relu2" => new UnaryMlpActivation("Relu", squared: true),
        "xielu" => new XieluActivation(hp.XieluBeta, hp.XieluEps),
        "swiglu_oai" => new SwigluOaiActivation(hp.SwigluAlpha, hp.SwigluLimit),
        _ => throw new NotSupportedException($"decoder: unknown mlp.activation '{hp.Activation}'."),
    };
}
