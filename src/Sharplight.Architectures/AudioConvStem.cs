using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The convolution stem of speech encoders over frame-level audio features or the raw waveform: 1-D convolutions along time (each followed
/// by an activation and optionally a group norm over the whole signal), then the result as rows [frames out, H], plus learned positions when
/// the model has them. Whisper: two kernel-3 convolutions with strides 1 and 2 and GELU, 3000 log-mel frames to 1500 rows. Moonshine: three
/// unpadded convolutions over the waveform (kernel 127 / stride 64 with tanh and a one-group norm, then 7 / 3 and 3 / 2 with GELU).
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>{arch}.</c>): <c>encoder.conv.strides</c> (one per convolution), optional <c>encoder.conv.padding</c> (zeros on both
/// sides, one per convolution; default kernel / 2), optional <c>encoder.conv.activations</c> (one per convolution) or
/// <c>encoder.conv.activation</c> (all of them; default "gelu"), optional <c>encoder.conv.norm_eps</c> (default 1e-5). Weights:
/// <c>encoder.conv.{i}.weight</c> [O, C, K] with optional <c>.bias</c> [O]; optional <c>encoder.conv.{i}.norm.weight</c> and <c>.bias</c> [O]
/// (a group norm with one group after the activation: every value of the signal normalized together, then a scale and shift per channel);
/// optional <c>encoder.position_embedding.weight</c> [rows, H]. The convolutions are the generic time-major <c>Conv1d</c> op; f16 and bf16
/// kernels are widened to f32 when loaded.
/// </remarks>
internal sealed class AudioConvStem
{
    private readonly long[] _strides;
    private readonly long[] _padding;
    private readonly long[] _kernels;
    private readonly string[] _activations;
    private readonly float _normEps;

    /// <summary>Reads the stem's hyper-parameters and adds its kernel views to <paramref name="weights"/>.</summary>
    /// <param name="src">The model source.</param>
    /// <param name="weights">Receives the f32 [O, C, K] kernels <c>encoder.conv.{i}.kernel</c>.</param>
    public AudioConvStem(ModelSource src, IDictionary<string, WeightInfo> weights)
    {
        _strides = src.Arch<long[]>("encoder.conv.strides");
        _kernels = new long[_strides.Length];
        string activation = src.Arch("encoder.conv.activation", "gelu");
        _activations = src.Arch("encoder.conv.activations", Enumerable.Repeat(activation, _strides.Length).ToArray());
        _normEps = src.Arch("encoder.conv.norm_eps", 1e-5f);
        for (int i = 0; i < _strides.Length; i++)
        {
            string p = $"encoder.conv.{i}.";
            var w = src.Weight(p + "weight") ?? throw new NotSupportedException($"{src.Architecture}: missing '{p}weight'.");
            if (w.Shape.Rank != 3)
            {
                throw new NotSupportedException($"{src.Architecture}: '{p}weight' must be [out, in, kernel], got {w.Shape}.");
            }

            _kernels[i] = w.Shape[2].Fixed;
            weights[p + "kernel"] = w.Type == DataType.F32 ? w.As(p + "kernel", w.Shape) : WeightConversions.ToF32(w, p + "kernel", w.Shape);
        }

        _padding = src.Arch("encoder.conv.padding", _kernels.Select(k => k / 2).ToArray());
        if (_padding.Length != _strides.Length || _activations.Length != _strides.Length)
        {
            throw new NotSupportedException($"{src.Architecture}: encoder.conv.padding and encoder.conv.activations need one entry per convolution.");
        }
    }

    /// <summary>The rows the stem makes of <paramref name="frames"/> input frames (fixed or symbolic).</summary>
    /// <param name="frames">The input frames.</param>
    public Dim Rows(Dim frames)
    {
        for (int i = 0; i < _strides.Length; i++)
        {
            long extra = (2 * _padding[i]) - _kernels[i];
            var padded = extra >= 0 ? frames + extra : frames - (-extra);
            frames = (_strides[i] == 1 ? padded : padded.FloorDiv(_strides[i])) + 1;
        }

        return frames;
    }

    /// <summary>Emits the stem over time-major <paramref name="signal"/> [F, C] and returns the rows [<see cref="Rows"/>(F), H].</summary>
    /// <param name="b">The graph builder (the kernel views must be resolvable through it).</param>
    /// <param name="signal">The audio features or waveform, time-major [frames, channels].</param>
    /// <param name="activation">Emits the activation of the given name.</param>
    public Value Build(GraphBuilder b, Value signal, Func<Value, string, Value> activation)
    {
        var x = signal;
        using (b.Group("encoder.conv"))
        {
            for (int i = 0; i < _strides.Length; i++)
            {
                string p = $"encoder.conv.{i}.";
                var kernel = b.Weight(p + "kernel");
                Value[] inputs = b.HasWeight(p + "bias") ? [x, kernel, b.Weight(p + "bias")] : [x, kernel];
                var attributes = Attributes.From(("stride", _strides[i]), ("pad_left", _padding[i]), ("pad_right", _padding[i]));
                x = b.Emit1("Conv1d", inputs, attributes, $"conv{i}");
                x = activation(x, _activations[i]);
                if (b.HasWeight(p + "norm.weight"))
                {
                    x = SignalNorm(b, x, p + "norm");
                }
            }
        }

        if (!b.HasWeight("encoder.position_embedding.weight"))
        {
            return x;
        }

        using (b.Group("input"))
        {
            return b.Add(x, b.Weight("encoder.position_embedding.weight"), "inp_embd");
        }
    }

    // A group norm with one group over the time-major signal [T, C]: mean and variance over every value, then a scale and shift per channel.
    private Value SignalNorm(GraphBuilder b, Value x, string module)
    {
        var all = Attributes.From(("mode", "mean"), ("axes", new long[] { 0, 1 }), ("keepdims", true));
        var centered = b.Sub(x, b.Emit1("Reduce", [x], all));
        var variance = b.Emit1("Reduce", [b.Unary("Square", centered)], all);
        var normed = b.Mul(centered, b.Unary("Rsqrt", b.Scale(variance, 1f, _normEps)));
        return b.Add(b.Mul(normed, b.Weight(module + ".weight")), b.Weight(module + ".bias"), module);
    }
}
