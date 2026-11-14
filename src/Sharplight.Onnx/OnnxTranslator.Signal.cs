using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Signal processing: DFT and STFT. A transform of moderate length is a MatMul against a constant basis (the STFT window folded in),
// so it runs on every provider through the matmul kernels; longer transforms use the IR Dft op (an FFT; CPU kernel).
internal sealed partial class OnnxTranslator
{
    /// <summary>Largest dense DFT basis (elements) emitted as a MatMul weight; longer transforms use the Dft op.</summary>
    private const long MaxDftBasisElements = 1L << 20;

    private void Dft(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        bool opset20 = Opset >= 20;
        long axis = opset20 ? OptConst(ins, 2, "DFT axis")?.ScalarInt() ?? -2 : n.Int("axis", 1);
        long? length = OptConst(ins, 1, "DFT dft_length")?.ScalarInt();
        int ax = NormalizeAxis(axis, x.Rank);
        if (ax < 0 || ax >= x.Rank - 1)
        {
            throw new InvalidDataException($"DFT axis {axis} is outside the signal dimensions of a rank-{x.Rank} input");
        }

        var spec = DftSpec.Create(x.Shape[ax].Fixed, x.Shape[-1].Fixed, length, n.Int("inverse", 0) != 0, n.Int("onesided", 0) != 0);
        SetDyn(n, 0, TransformAxis(x, ax, spec, null, null, Nm(n)));
    }

    private void Stft(OnnxNode n, TranslatedTensor?[] ins)
    {
        var signal = X(ins, 0);
        if (signal.Rank != 3)
        {
            throw new InvalidDataException($"STFT expects a signal [batch, length, 1|2], got {signal.Shape}");
        }

        long step = ConstIn(ins, 1, "STFT frame_step").ScalarInt();
        var windowIn = Opt(ins, 2);
        long? frameLength = OptConst(ins, 3, "STFT frame_length")?.ScalarInt();
        long frame = windowIn is not null ? (windowIn.IsConst ? windowIn.C!.Count : windowIn.V!.Shape.Elements()) : frameLength
            ?? throw new InvalidDataException("STFT needs a window or a frame_length");
        if (frameLength is { } fl && fl != frame)
        {
            throw new InvalidDataException($"STFT frame_length {fl} differs from the window length {frame}");
        }

        long signalLength = signal.Shape[1].Fixed;
        long frames = ((signalLength - frame) / step) + 1;
        if (step <= 0 || frame <= 0 || frames <= 0)
        {
            throw new InvalidDataException($"STFT of {signalLength} samples with frame {frame} and step {step} has no frames");
        }

        // Framing: one gather along the signal axis builds [batch, frames, frame, components].
        var index = new int[frames * frame];
        for (long f = 0; f < frames; f++)
        {
            for (long k = 0; k < frame; k++)
            {
                index[(f * frame) + k] = (int)((f * step) + k);
            }
        }

        string name = Nm(n);
        var indices = _b.Constant(name + ".frames", DataType.I32, new Shape(frames, frame), IntBytes(index));
        var framed = Emit("GatherAxis", [signal, indices], Attributes.From(("axis", 1L)), name + ".framed");
        var spec = DftSpec.Create(frame, signal.Shape[2].Fixed, null, inverse: false, oneSided: n.Int("onesided", 1) != 0);
        double[]? window = windowIn?.C?.Floats();
        Value? runtimeWindow = windowIn is { IsConst: false } ? Rs(windowIn.V!, frame, 1) : null;
        SetDyn(n, 0, TransformAxis(framed, 2, spec, window, runtimeWindow, name));
    }

    /// <summary>
    /// DFT along axis <paramref name="ax"/> of x <c>[..., N, ..., C]</c> (C = last axis): moves the axis next to the components, multiplies
    /// by the basis (or runs the Dft op) and moves it back.
    /// </summary>
    private Value TransformAxis(Value x, int ax, DftSpec spec, double[]? window, Value? runtimeWindow, string name)
    {
        int rank = x.Rank;
        var perm = Enumerable.Range(0, rank).Where(d => d != ax && d != rank - 1).Append(ax).Append(rank - 1).ToArray();
        var inverse = new int[rank];
        for (int d = 0; d < rank; d++)
        {
            inverse[perm[d]] = d;
        }

        var moved = Permute(x, perm);
        if (runtimeWindow is not null)
        {
            moved = Emit("Mul", [moved, runtimeWindow], null, name + ".windowed");
        }

        var lead = Fixed(moved)[..^2];
        Value y;
        if ((long)spec.OutBins * spec.OutComponents * spec.InLength * spec.InComponents <= MaxDftBasisElements)
        {
            var basis = OnnxConst.FromBytes(OnnxType.Float, [spec.OutBins * spec.OutComponents, spec.InLength * spec.InComponents],
                ToBytes(SignalMath.Basis(spec, window)), name + ".basis");
            var rows = Rs(moved, [.. lead, spec.InLength * spec.InComponents]);
            y = Emit("MatMul", [rows, WeightValue(basis, keepHalf: false, name + ".basis")], null, name);
            Heavy();
            y = Rs(y, [.. lead, spec.OutBins, spec.OutComponents]);
        }
        else
        {
            if (window is not null)
            {
                var w = OnnxConst.FromFloats(OnnxType.Float, [spec.InLength, 1], window, name + ".window");
                moved = Emit("Mul", [moved, ConstValue(w, name + ".window")], null, name + ".windowed");
            }

            y = Emit("Dft", [moved], Attributes.From(("length", (long)spec.Length), ("out_bins", (long)spec.OutBins), ("inverse", spec.Inverse),
                ("real_output", spec.InverseOneSided)), name);
            Heavy();
        }

        return Permute(y, inverse);
    }

    private static byte[] IntBytes(int[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
