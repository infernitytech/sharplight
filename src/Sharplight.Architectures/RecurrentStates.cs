using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>Factory for the f32 recurrent state declarations shared by the convolution and state-space mixers.</summary>
internal static class RecurrentStates
{
    /// <summary>Causal-convolution window state <c>{group}.conv_state</c> of shape [<paramref name="window"/>, <paramref name="channels"/>].</summary>
    public static StateDefinition ConvWindow(string group, int window, int channels)
        => Recurrent($"{group}.conv_state", group, new Shape(window, channels));

    /// <summary>Selective-scan state <c>{group}.ssm_state</c> of shape [<paramref name="heads"/>, <paramref name="headDim"/>, <paramref name="stateSize"/>].</summary>
    public static StateDefinition Scan(string group, long heads, long headDim, long stateSize)
        => Recurrent($"{group}.ssm_state", group, new Shape(heads, headDim, stateSize));

    private static StateDefinition Recurrent(string name, string group, Shape shape)
        => new() { Name = name, Type = DataType.F32, Shape = shape, Kind = StateKind.Recurrent, Group = group };
}
