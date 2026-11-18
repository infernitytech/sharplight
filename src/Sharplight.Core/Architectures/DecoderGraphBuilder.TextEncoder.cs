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
    /// <param name="f32Activations">Whether the matrix products keep f32 activations (<see cref="GraphBuilder.F32MatMulActivations"/>).</param>
    /// <exception cref="InvalidDataException">The decoder's layout has no single hidden state per layer.</exception>
    public ComputeGraph BuildLayerHiddenStates(string name, string tokensInput, string hiddenOutput, int rows, int layer, bool finalNorm = false,
        bool f32Activations = false)
        => BuildLayerHiddenStates(name, tokensInput, hiddenOutput, rows, [layer], finalNorm, f32Activations: f32Activations);

    /// <summary>
    /// The hidden states entering several layers, concatenated per token in the given order (FLUX.2: Qwen3's <c>hidden_states</c> 9, 18
    /// and 27 → [rows, 3 · hidden size]); see the single-layer overload.
    /// </summary>
    /// <param name="name">Graph name.</param>
    /// <param name="tokensInput">Name of the token-id input i32 [rows].</param>
    /// <param name="hiddenOutput">Name of the output f32 [rows, layers · hidden size].</param>
    /// <param name="rows">Tokens per prompt.</param>
    /// <param name="layers">The layers whose inputs are concatenated (0 … layer count).</param>
    /// <param name="finalNorm">Whether the last layer's output (layer = the layer count) goes through the final norm.</param>
    /// <param name="maskInput">
    /// The name of an f32 [rows] input marking the prompt's tokens (1) and the padding (0), or null. With it the padding rows attend
    /// only to the prompt's tokens, as transformers' padding mask makes them (FLUX.2 reads every row, the padding's included); without
    /// it padding rows also see the padding before them (the prompt's rows are the same either way).
    /// </param>
    /// <param name="f32Activations">Whether the matrix products keep f32 activations (<see cref="GraphBuilder.F32MatMulActivations"/>).</param>
    /// <param name="imageRows">
    /// Whether the graph takes image rows of a vision-language model (Qwen-Image-Edit's Qwen2.5-VL with the reference images in the prompt):
    /// inputs <c>positions</c> i32 [rows, position axes] (M-RoPE positions instead of the row index), <c>embeddings</c> f32 [rows, hidden size]
    /// and <c>embedding_mask</c> f32 [rows, 1]; rows with mask 1 take their embedding instead of the token's.
    /// </param>
    /// <exception cref="InvalidDataException">The decoder's layout has no single hidden state per layer.</exception>
    public ComputeGraph BuildLayerHiddenStates(string name, string tokensInput, string hiddenOutput, int rows, IReadOnlyList<int> layers, bool finalNorm = false,
        string? maskInput = null, bool f32Activations = false, bool imageRows = false)
    {
        if (layers.Count == 0 || layers.Any(l => l < 0 || l > _hp.LogicalLayers) || _hp.AltUpStreams > 0 || _hp.Cycles is not null)
        {
            throw new InvalidDataException($"decoder: cannot output the hidden states before layers {string.Join(", ", layers)}.");
        }

        var b = new GraphBuilder(name, _src.Weight) { F32MatMulActivations = f32Activations };
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

        var positions = imageRows
            ? b.Input("positions", InputRole.Other, DataType.I32, new Shape(rows, _hp.PositionAxes))
            : b.Constant("prompt_positions", DataType.I32, new Shape(rows, _hp.PositionAxes), MemoryMarshal.AsBytes(positionValues.AsSpan()).ToArray());
        var spans = b.Constant("prompt_spans", DataType.I32, new Shape(rows, 2), MemoryMarshal.AsBytes(spanValues.AsSpan()).ToArray());
        if (maskInput is not null)
        {
            // Every row's span is the prompt's tokens [0, n) (with the causal limit): spans (0, n) from the mask's sum.
            var mask = b.Input(maskInput, InputRole.Other, DataType.F32, new Shape(rows));
            var count = b.Reshape(b.Emit1("ReduceSum", [mask]), 1, 1);
            var second = new float[rows * 2];
            for (int row = 0; row < rows; row++)
            {
                second[row * 2 + 1] = 1f;
            }

            var selector = b.Constant("prompt_span_selector", DataType.F32, new Shape(rows, 2), MemoryMarshal.AsBytes(second.AsSpan()).ToArray());
            spans = b.Emit1("Cast", [b.Mul(selector, count)], Attributes.From(("type", "i32")));
        }
        Value x;
        Value? perLayer = null;
        using (b.Group("input"))
        {
            x = TokenEmbeddings(b, RouteAdapters(b, tokens, spans));
            if (imageRows)
            {
                var embeddings = b.Input("embeddings", InputRole.Other, DataType.F32, new Shape(rows, _hp.NEmbd));
                var rowMask = b.Input("embedding_mask", InputRole.Other, DataType.F32, new Shape(rows, 1));
                x = b.Add(b.Mul(x, b.Scale(rowMask, -1f, 1f)), b.Mul(embeddings, rowMask), "inp_mixed");
            }

            x = AddPositionEmbeddings(b, x, positions);
            if (_hp.PerLayerSize > 0)
            {
                perLayer = PerLayerInputs(b, x, tokens);
            }
        }

        _tapValues = [];
        x = Layers(b, x, positions, FrequencyFactors(b), perLayer, spans);
        _tapValues[_hp.LogicalLayers] = finalNorm ? OutputNorm(b, x) : x;
        b.Output(hiddenOutput, layers.Count == 1 ? _tapValues[layers[0]] : b.Concat(1, [.. layers.Select(l => _tapValues[l])]));
        _tapValues = null;
        _adapterSlots = null;
        return b.Build();
    }
}
