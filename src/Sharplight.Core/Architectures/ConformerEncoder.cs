using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The FastConformer speech encoder of NVIDIA's NeMo models (Parakeet, Canary) as generic ops: a subsampling stack of 2-D convolutions
/// over the time-major log-mel features (<c>SequenceConv2d</c>, the time axis variable), a linear projection of the flattened channels,
/// then conformer blocks — a half-step feed-forward, self-attention with Transformer-XL relative positions (<c>RelativeAttention</c>), a
/// convolution module (pointwise projection, GLU, depthwise <c>Conv1d</c>, batch norm, activation, pointwise projection) and a second
/// half-step feed-forward, each pre-norm with LayerNorm, and a LayerNorm after the block.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>{arch}.</c>): <c>hidden_size</c> (of the encoder, or of the decoder when <c>encoder.hidden_size</c> gives the encoder's), <c>encoder.layer_count</c>, <c>encoder.attention.head_count</c>,
/// <c>encoder.subsampling.strides</c>, <c>encoder.subsampling.groups</c> and <c>encoder.subsampling.activate</c> (one entry per convolution
/// of the stack: time and feature stride, groups, whether a ReLU follows), optional <c>encoder.input_scale</c> (factor after the projection,
/// default 1), <c>encoder.position.base</c> (default 10000), <c>encoder.mlp.activation</c> (default "silu", also the convolution module's),
/// <c>encoder.norm_eps</c> (default 1e-5), <c>encoder.conv.norm_eps</c> (batch norm, default 1e-5).
/// Streaming models (Nemotron ASR streaming) add <c>encoder.causal</c> (bool: the subsampling convolutions pad (kernel − 1, stride − 1) on both axes
/// and the depthwise convolution only to the left), <c>encoder.conv.norm</c> ("batch" (default) or "layer": a LayerNorm with <c>conv.norm.{weight,bias}</c>
/// instead of the folded batch norm) and <c>encoder.attention.chunked</c> (bool: an i32 <see cref="InputRole.RowSpans"/> input [rows, 2] limits every
/// row's attention, see <see cref="RowSpansInput"/>).
/// <para>Weights: <c>encoder.subsampling.{i}.weight</c> [O, C/groups, KT, KW] (padding (K − 1) / 2 on both axes) with optional
/// <c>.bias</c>; <c>encoder.subsampling.out.weight</c> [H, C · W'] and <c>.bias</c>; per block <c>encoder.layers.{i}.</c>:
/// <c>ff1_norm</c>, <c>ff1.up</c>, <c>ff1.down</c>, <c>attention_norm</c>, <c>attention.{q,k,v,o}</c>, <c>attention.position.weight</c>
/// [H, H] (the projection of the relative position embeddings), <c>attention.bias_u</c> and <c>attention.bias_v</c> [heads, head_dim],
/// <c>conv_norm</c>, <c>conv.pointwise1</c> [2H, H], <c>conv.depthwise.weight</c> [H, 1, K], <c>conv.norm.{weight,bias,mean,variance}</c>
/// (batch norm statistics, folded into a scale and shift when loaded), <c>conv.pointwise2</c> [H, H], <c>ff2_norm</c>, <c>ff2.up</c>,
/// <c>ff2.down</c>, <c>output_norm</c>; projections and norms may have a <c>.bias</c>.</para>
/// <para>Inputs of the graph it builds: the features [<see cref="WellKnown.AudioFrameSymbol"/>, mel bins] (<see cref="InputRole.AudioFeatures"/>)
/// and the relative distances [2·rows − 1, 1] (<see cref="InputRole.Positions"/>), rows − 1 down to −(rows − 1).</para>
/// </remarks>
internal sealed class ConformerEncoder
{
    /// <summary>The name of the optional attention spans input [rows, 2] (first key, key count of every row).</summary>
    public const string RowSpansInput = "row_spans";

    private readonly ModelSource _src;
    private readonly int _hidden;
    private readonly int _layers;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _melBins;
    private readonly long[] _strides;
    private readonly long[] _groups;
    private readonly bool[] _activate;
    private readonly long[] _kernels;
    private readonly float _inputScale;
    private readonly float _eps;
    private readonly string _activation;
    private readonly bool _causal;
    private readonly bool _layerNorm;
    private readonly bool _chunked;
    private readonly float[] _frequencies;

    /// <summary>Reads the hyper-parameters and adds the derived weights (f32 kernels, folded batch norms) to <paramref name="weights"/>.</summary>
    /// <param name="src">The model source.</param>
    /// <param name="weights">Receives the derived weights.</param>
    public ConformerEncoder(ModelSource src, IDictionary<string, WeightInfo> weights)
    {
        _src = src;
        _hidden = src.Arch("encoder.hidden_size", src.Arch<int>("hidden_size"));
        _layers = src.Arch<int>("encoder.layer_count");
        _heads = src.Arch<int>("encoder.attention.head_count");
        _headDim = _hidden / _heads;
        _melBins = src.Arch<int>("audio.mel_bins");
        _strides = src.Arch<long[]>("encoder.subsampling.strides");
        _groups = src.Arch<long[]>("encoder.subsampling.groups");
        _activate = src.Arch<bool[]>("encoder.subsampling.activate");
        if (_groups.Length != _strides.Length || _activate.Length != _strides.Length)
        {
            throw new NotSupportedException($"{src.Architecture}: encoder.subsampling.* need one entry per convolution.");
        }

        _inputScale = src.Arch("encoder.input_scale", 1f);
        _eps = src.Arch("encoder.norm_eps", 1e-5f);
        _activation = src.Arch("encoder.mlp.activation", "silu");
        _causal = src.Arch("encoder.causal", false);
        _layerNorm = src.Arch("encoder.conv.norm", "batch") == "layer";
        _chunked = src.Arch("encoder.attention.chunked", false);
        _kernels = new long[_strides.Length];
        for (int i = 0; i < _strides.Length; i++)
        {
            string name = $"encoder.subsampling.{i}.weight";
            var w = src.Weight(name) ?? throw new NotSupportedException($"{src.Architecture}: missing '{name}'.");
            _kernels[i] = w.Shape[2].Fixed;
            weights[$"encoder.subsampling.{i}.kernel"] = ToF32(w, $"encoder.subsampling.{i}.kernel");
        }

        float convEps = src.Arch("encoder.conv.norm_eps", 1e-5f);
        for (int il = 0; il < _layers; il++)
        {
            string p = $"encoder.layers.{il}.conv.";
            var depthwise = src.Weight(p + "depthwise.weight") ?? throw new NotSupportedException($"{src.Architecture}: missing '{p}depthwise.weight'.");
            weights[p + "depthwise.kernel"] = ToF32(depthwise, p + "depthwise.kernel");
            if (!_layerNorm)
            {
                AddFoldedNorm(src, weights, p + "norm", convEps);
            }
        }

        double positionBase = src.Arch("encoder.position.base", 10000.0);
        _frequencies = new float[_hidden / 2];
        for (int i = 0; i < _frequencies.Length; i++)
        {
            _frequencies[i] = (float)(1.0 / Math.Pow(positionBase, 2.0 * i / _hidden));
        }
    }

    /// <summary>Encoder hidden size.</summary>
    public int Hidden => _hidden;

    /// <summary>The rows the subsampling makes of <paramref name="frames"/> feature frames (fixed or symbolic).</summary>
    /// <param name="frames">The feature frames.</param>
    public Dim Rows(Dim frames)
    {
        for (int i = 0; i < _strides.Length; i++)
        {
            if (_strides[i] == 1)
            {
                continue;
            }

            long extra = _causal ? (_kernels[i] - 1) + (_strides[i] - 1) - _kernels[i] : (2 * ((_kernels[i] - 1) / 2)) - _kernels[i];
            var padded = extra >= 0 ? frames + extra : frames - (-extra);
            frames = padded.FloorDiv(_strides[i]) + 1;
        }

        return frames;
    }

    /// <summary>Declares the inputs and emits the encoder; returns its output rows [rows, H].</summary>
    /// <param name="b">The graph builder (the derived weights must be resolvable through it).</param>
    public Value Build(GraphBuilder b)
    {
        Value x;
        Value distances;
        Value? spans = null;
        using (b.Group("input"))
        {
            var features = b.Input("features", InputRole.AudioFeatures, DataType.F32, new Shape(Dim.Sym(WellKnown.AudioFrameSymbol), _melBins));
            x = Subsampling(b, features);
            var rows = x.Shape[0];
            distances = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(rows + rows - 1, 1));
            if (_chunked)
            {
                spans = b.Input(RowSpansInput, InputRole.RowSpans, DataType.I32, new Shape(rows, 2));
            }
        }

        var positions = PositionEmbeddings(b, distances);
        float scale = 1f / MathF.Sqrt(_headDim);
        for (int il = 0; il < _layers; il++)
        {
            using var _ = b.Group($"encoder.layers.{il}");
            string p = $"encoder.layers.{il}.";
            x = b.Add(x, b.Scale(FeedForward(b, Norm(b, x, p + "ff1_norm"), p + "ff1."), 0.5f));
            var h = Norm(b, x, p + "attention_norm");
            var q = Heads(b, Linear(b, h, p + "attention.q"));
            var k = Heads(b, Linear(b, h, p + "attention.k"));
            var v = Heads(b, Linear(b, h, p + "attention.v"));
            var relative = b.Reshape(b.Linear(positions, p + "attention.position.weight"), 0, _heads, _headDim);
            var content = b.Add(q, b.Weight(p + "attention.bias_u"));
            var position = b.Add(q, b.Weight(p + "attention.bias_v"));
            Value[] attentionInputs = spans is null ? [content, position, k, v, relative] : [content, position, k, v, relative, spans];
            var attention = b.Emit1("RelativeAttention", attentionInputs, Attributes.From(("scale", scale)), $"{p}attn");
            x = b.Add(x, Linear(b, b.Reshape(attention, 0, _hidden), p + "attention.o", $"{p}attn_out"));
            x = b.Add(x, ConvolutionModule(b, Norm(b, x, p + "conv_norm"), p + "conv."));
            x = b.Add(x, b.Scale(FeedForward(b, Norm(b, x, p + "ff2_norm"), p + "ff2."), 0.5f));
            x = Norm(b, x, p + "output_norm", $"{p}out");
        }

        return x;
    }

    // [F, mels] → [F, 1, mels] → the convolution stack → [rows, C, W'] → [rows, C · W'] → the projection [rows, H].
    private Value Subsampling(GraphBuilder b, Value features)
    {
        var x = b.Reshape(features, 0, 1, _melBins);
        for (int i = 0; i < _strides.Length; i++)
        {
            string p = $"encoder.subsampling.{i}.";
            long pad = (_kernels[i] - 1) / 2;
            long before = _causal ? _kernels[i] - 1 : pad;
            long after = _causal ? _strides[i] - 1 : pad;
            Value[] inputs = b.HasWeight(p + "bias") ? [x, b.Weight(p + "kernel"), b.Weight(p + "bias")] : [x, b.Weight(p + "kernel")];
            var attributes = Attributes.From(("strides", new[] { _strides[i], _strides[i] }), ("pads", new[] { before, before, after, after }), ("groups", _groups[i]));
            x = b.Emit1("SequenceConv2d", inputs, attributes, $"subsampling{i}");
            if (_activate[i])
            {
                x = b.Unary("Relu", x);
            }
        }

        x = b.Reshape(x, 0, x.Shape[1].Fixed * x.Shape[2].Fixed);
        x = b.Linear(x, "encoder.subsampling.out.weight", b.HasWeight("encoder.subsampling.out.bias") ? "encoder.subsampling.out.bias" : null, "subsampling_out");
        return _inputScale != 1f ? b.Scale(x, _inputScale) : x;
    }

    // Sinusoids of the distances, sine and cosine interleaved per frequency: [P, H].
    private Value PositionEmbeddings(GraphBuilder b, Value distances)
    {
        var bytes = new byte[_frequencies.Length * sizeof(float)];
        Buffer.BlockCopy(_frequencies, 0, bytes, 0, bytes.Length);
        var frequencies = b.Constant("encoder.position.frequencies", DataType.F32, new Shape(1, _frequencies.Length), bytes);
        var angles = b.Mul(b.Emit1("Cast", [distances], Attributes.From(("type", "f32"))), frequencies);
        int half = _frequencies.Length;
        var sine = b.Reshape(b.Unary("Sin", angles), 0, half, 1);
        var cosine = b.Reshape(b.Unary("Cos", angles), 0, half, 1);
        return b.Reshape(b.Concat(2, sine, cosine), 0, _hidden);
    }

    private Value ConvolutionModule(GraphBuilder b, Value h, string p)
    {
        var projected = Linear(b, h, p + "pointwise1");
        var glu = b.Mul(b.Slice(projected, 1, 0, _hidden), b.Unary("Sigmoid", b.Slice(projected, 1, _hidden, _hidden)));
        var kernel = b.Weight(p + "depthwise.kernel");
        long pad = (kernel.Shape[2].Fixed - 1) / 2;
        long left = _causal ? kernel.Shape[2].Fixed - 1 : pad;
        long right = _causal ? 0 : pad;
        Value[] inputs = b.HasWeight(p + "depthwise.bias") ? [glu, kernel, b.Weight(p + "depthwise.bias")] : [glu, kernel];
        var conv = b.Emit1("Conv1d", inputs, Attributes.From(("pad_left", left), ("pad_right", right), ("groups", (long)_hidden)), p + "depthwise");
        var normed = _layerNorm ? Norm(b, conv, p + "norm") : b.Add(b.Mul(conv, b.Weight(p + "norm.scale")), b.Weight(p + "norm.shift"));
        return Linear(b, Activation(b, normed), p + "pointwise2");
    }

    private Value FeedForward(GraphBuilder b, Value h, string p) => Linear(b, Activation(b, Linear(b, h, p + "up")), p + "down");

    private Value Activation(GraphBuilder b, Value x) => _activation switch
    {
        "silu" => b.Silu(x),
        "relu" => b.Unary("Relu", x),
        "gelu" => b.Unary("Gelu", x),
        var other => throw new NotSupportedException($"{_src.Architecture}: unknown encoder activation '{other}'."),
    };

    private Value Heads(GraphBuilder b, Value v) => b.Reshape(v, 0, _heads, _headDim);

    private static Value Linear(GraphBuilder b, Value v, string module, string? name = null)
        => b.Linear(v, module + ".weight", b.HasWeight(module + ".bias") ? module + ".bias" : null, name);

    private Value Norm(GraphBuilder b, Value v, string module, string? name = null)
        => b.Emit1("LayerNorm", b.HasWeight(module + ".bias") ? [v, b.Weight(module + ".weight"), b.Weight(module + ".bias")] : [v, b.Weight(module + ".weight")],
            Attributes.From(("eps", _eps)), name);

    private static WeightInfo ToF32(WeightInfo w, string name) => w.Type == DataType.F32 ? w.As(name, w.Shape) : WeightConversions.ToF32(w, name, w.Shape);

    // Batch norm with running statistics as one scale and shift per channel: y = x · w / √(var + eps) + (b − mean · w / √(var + eps)).
    private static void AddFoldedNorm(ModelSource src, IDictionary<string, WeightInfo> weights, string module, float eps)
    {
        WeightInfo Part(string name) => src.Weight($"{module}.{name}") ?? throw new NotSupportedException($"{src.Architecture}: missing '{module}.{name}'.");
        var weight = Part("weight");
        var bias = Part("bias");
        var mean = Part("mean");
        var variance = Part("variance");
        var shape = weight.Shape;
        float[] Scale()
        {
            var w = WeightConversions.ReadFloats(weight);
            var v = WeightConversions.ReadFloats(variance);
            var result = new float[w.Length];
            for (int c = 0; c < w.Length; c++)
            {
                result[c] = (float)(w[c] / Math.Sqrt(v[c] + (double)eps));
            }

            return result;
        }

        weights[module + ".scale"] = WeightInfo.Derived(module + ".scale", DataType.F32, shape, () => Bytes(Scale()));
        weights[module + ".shift"] = WeightInfo.Derived(module + ".shift", DataType.F32, shape, () =>
        {
            var scale = Scale();
            var b = WeightConversions.ReadFloats(bias);
            var m = WeightConversions.ReadFloats(mean);
            var result = new float[scale.Length];
            for (int c = 0; c < scale.Length; c++)
            {
                result[c] = b[c] - (m[c] * scale[c]);
            }

            return Bytes(result);
        });
    }

    private static byte[] Bytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
