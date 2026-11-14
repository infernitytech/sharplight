using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>Factory for the recurrent state declarations shared by the convolution and state-space mixers (f32 unless stated).</summary>
internal static class RecurrentStates
{
    /// <summary>Causal-convolution window state <c>{group}.conv_state</c> of shape [<paramref name="window"/>, <paramref name="channels"/>].</summary>
    public static StateDefinition ConvWindow(string group, int window, int channels)
        => Recurrent($"{group}.conv_state", group, new Shape(window, channels));

    /// <summary>
    /// Selective-scan state <c>{group}.ssm_state</c> of shape [<paramref name="heads"/>, <paramref name="headDim"/>, <paramref name="stateSize"/>],
    /// of element type <paramref name="type"/> (default f32; 16-bit types only for ops that support them).
    /// </summary>
    public static StateDefinition Scan(string group, long heads, long headDim, long stateSize, DataType? type = null)
        => Recurrent($"{group}.ssm_state", group, new Shape(heads, headDim, stateSize), type ?? DataType.F32);

    private static StateDefinition Recurrent(string name, string group, Shape shape, DataType? type = null)
        => new() { Name = name, Type = type ?? DataType.F32, Shape = shape, Kind = StateKind.Recurrent, Group = group };
}
