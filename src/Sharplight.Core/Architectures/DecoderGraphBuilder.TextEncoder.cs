using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

internal sealed partial class DecoderGraphBuilder
{
    /// <summary>
    /// A text-encoder graph over one padded prompt of <paramref name="rows"/> tokens (causal attention, no KV cache): the hidden state
    /// that enters layer <paramref name="layer"/> (transformers' <c>hidden_states[layer]</c>; the layer count gives the last layer's output
    /// before the final norm), as image generators use a language model's hidden states for their conditioning (Z-Image: Qwen3's
    /// <c>hidden_states[-2]</c>). Padding after the prompt does not reach the prompt's rows (causal attention).
    /// </summary>
    /// <param name="name">Graph name.</param>
    /// <param name="tokensInput">Name of the token-id input i32 [rows].</param>
    /// <param name="hiddenOutput">Name of the output f32 [rows, hidden size].</param>
    /// <param name="rows">Tokens per prompt.</param>
    /// <param name="layer">The layer whose input is the output (0 … layer count).</param>
    /// <param name="finalNorm">
    /// Whether the last layer's output goes through the final norm (transformers' <c>hidden_states[-1]</c>, Qwen-Image's conditioning);
    /// only with <paramref name="layer"/> = the layer count.
    /// </param>
    /// <exception cref="InvalidDataException">The decoder's layout has no single hidden state per layer.</exception>
    public ComputeGraph BuildLayerHiddenStates(string name, string tokensInput, string hiddenOutput, int rows, int layer, bool finalNorm = false)
    {
        if (layer < 0 || layer > _hp.LogicalLayers || _hp.AltUpStreams > 0 || _hp.Cycles is not null)
        {
            throw new InvalidDataException($"decoder: cannot output the hidden state before layer {layer}.");
        }

        var b = new GraphBuilder(name, _src.Weight);
        var tokens = b.Input(tokensInput, InputRole.TokenIds, DataType.I32, new Shape(rows));
        var positionValues = new int[rows * _hp.PositionAxes];
        var spanValues = new int[rows * 2];
        for (int row = 0; row < rows; row++)
        {
            for (int axis = 0; axis < _hp.PositionAxes; axis++)
            {
                positionValues[row * _hp.PositionAxes + axis] = row;
            }

            spanValues[row * 2 + 1] = rows;
        }

        var positions = b.Constant("prompt_positions", DataType.I32, new Shape(rows, _hp.PositionAxes), MemoryMarshal.AsBytes(positionValues.AsSpan()).ToArray());
        var spans = b.Constant("prompt_spans", DataType.I32, new Shape(rows, 2), MemoryMarshal.AsBytes(spanValues.AsSpan()).ToArray());
        Value x;
        Value? perLayer = null;
        using (b.Group("input"))
        {
            x = AddPositionEmbeddings(b, TokenEmbeddings(b, RouteAdapters(b, tokens, spans)), positions);
            if (_hp.PerLayerSize > 0)
            {
                perLayer = PerLayerInputs(b, x, tokens);
            }
        }

        _tapValues = [];
        x = Layers(b, x, positions, FrequencyFactors(b), perLayer, spans);
        _tapValues[_hp.LogicalLayers] = finalNorm ? OutputNorm(b, x) : x;
        b.Output(hiddenOutput, _tapValues[layer]);
        _tapValues = null;
        _adapterSlots = null;
        return b.Build();
    }
}
