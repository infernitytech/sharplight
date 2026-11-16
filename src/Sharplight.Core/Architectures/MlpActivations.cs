using Sharplight.Models;

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
        "xielu" => new XieluActivation(hp.XieluBeta, hp.XieluEps),
        "swiglu_oai" => new SwigluOaiActivation(hp.SwigluAlpha, hp.SwigluLimit),
        _ => Create(hp.Activation),
    };

    /// <summary>
    /// Creates the activation <paramref name="name"/>, reading the parameters of a parameterized one from <paramref name="source"/>:
    /// <c>situ</c> (<see cref="SituActivation"/>: <c>mlp.situ_beta</c>, default 1, and <c>mlp.situ_linear_beta</c>, default 0 = no cap on the up
    /// projection); the parameter-free names of <see cref="Create(string)"/> otherwise.
    /// </summary>
    /// <exception cref="NotSupportedException">The name is unknown.</exception>
    public static IMlpActivation Create(string name, ModelSource source) => name switch
    {
        "situ" => new SituActivation(source.Arch("mlp.situ_beta", 1f), source.Arch("mlp.situ_linear_beta", 0f)),
        _ => Create(name),
    };

    /// <summary>
    /// Creates a parameter-free activation. Names: <c>silu</c>, <c>gelu</c> (erf), <c>gelu_tanh</c>, <c>relu</c>, <c>relu2</c> (squared ReLU).
    /// </summary>
    /// <exception cref="NotSupportedException">The name is unknown or the activation has parameters.</exception>
    public static IMlpActivation Create(string name) => name switch
    {
        "silu" => new UnaryMlpActivation("Silu"),
        "gelu" => new UnaryMlpActivation("Gelu"),
        "gelu_tanh" => new UnaryMlpActivation("GeluTanh"),
        "relu" => new UnaryMlpActivation("Relu"),
        "relu2" => new UnaryMlpActivation("Relu", squared: true),
        _ => throw new NotSupportedException($"unknown mlp.activation '{name}'."),
    };
}
