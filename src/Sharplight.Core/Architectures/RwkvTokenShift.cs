using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// RWKV "token shift": the previous token's (normalized) input of a layer, carried across runs in a one-row recurrent state.
/// Implemented as a two-tap causal convolution with the fixed weights (1, 0), so it needs no op of its own and inherits the
/// multi-sequence form (per-slot state, a fresh sequence starts from zeros) of <c>CausalConv1d</c>.
/// </summary>
internal sealed class RwkvTokenShift
{
    private readonly int _channels;

    /// <summary>Creates the helper for rows of <paramref name="channels"/> channels.</summary>
    public RwkvTokenShift(int channels)
    {
        _channels = channels;
    }

    /// <summary>Declares the one-row state that carries the last input of a run.</summary>
    public StateDefinition DefineState(string name, string group)
        => new() { Name = name, Type = DataType.F32, Shape = new Shape(1, _channels), Kind = StateKind.Recurrent, Group = group };

    /// <summary>Adds the selector weights (1, 0) per channel to the graph; create once per graph and pass to <see cref="Previous"/>.</summary>
    public Value Selector(GraphBuilder b)
    {
        var bytes = new byte[_channels * 2 * sizeof(float)];
        for (int c = 0; c < _channels; c++)
        {
            BitConverter.TryWriteBytes(bytes.AsSpan(c * 2 * sizeof(float), sizeof(float)), 1f);
        }

        return b.Constant("rwkv_token_shift_selector", DataType.F32, new Shape(_channels, 2), bytes);
    }

    /// <summary>Row t of the result is row t-1 of <paramref name="x"/> (the state for t = 0); the state is advanced to the last row.</summary>
    public Value Previous(GraphBuilder b, Value x, Value selector, StateDefinition state)
    {
        var s = b.State(state);
        return b.Emit("CausalConv1d", [x, selector, s], Attributes.From(("activation", "none")), writes: [s])[0];
    }
}
