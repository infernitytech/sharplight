using Sharplight.Graph;

namespace Sharplight.Onnx;

// Convolution whose weights (or bias) are runtime tensors: the IR Conv packs constant weights at compile time, so a runtime
// weight is applied as im2col (a constant-index gather over the zero-padded input) followed by a batched matrix product.
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// Conv with runtime weights: cols[n, g, (c, kh, kw), p] gathers the padded input at constant offsets, and
    /// y[n, g, o, p] = w[g, o, (c, kh, kw)] · cols. 1-D and 2-D, any groups, strides, dilations and pads.
    /// </summary>
    private void ConvRuntimeWeight(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x0 = X(ins, 0);
        var w0 = AsValue(ins[1]!);
        bool oneD = x0.Rank == 3;
        if (w0.Rank != x0.Rank)
        {
            throw new NotSupportedException("only 1-D and 2-D convolutions are supported");
        }

        var x = To4D(x0);
        long outChannels = w0.Shape[0].Fixed;
        long perGroup = w0.Shape[1].Fixed;
        long kh = oneD ? 1 : w0.Shape[2].Fixed;
        long kw = w0.Shape[w0.Rank - 1].Fixed;
        long groups = n.Int("group", 1);
        long channels = x.Shape[1].Fixed;
        if (channels != groups * perGroup || outChannels % groups != 0)
        {
            throw new NotSupportedException($"Conv groups={groups} does not divide the channel counts");
        }

        var strides = ExpandSpatial(n.Ints("strides"), 1, oneD);
        var dilations = ExpandSpatial(n.Ints("dilations"), 1, oneD);
        var pads = ResolvePads(n, x, [kh, kw], strides, dilations, oneD);
        long hp = x.Shape[2].Fixed + pads[0] + pads[2];
        long wp = x.Shape[3].Fixed + pads[1] + pads[3];
        long ho = (hp - dilations[0] * (kh - 1) - 1) / strides[0] + 1;
        long wo = (wp - dilations[1] * (kw - 1) - 1) / strides[1] + 1;
        if (ho <= 0 || wo <= 0)
        {
            throw new NotSupportedException("Conv with an empty output");
        }

        string name = Nm(n);
        var padded = pads.All(p => p == 0) ? x : Emit("Pad", [x], Attributes.From(("pads", L(0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3])), ("mode", "constant"), ("value", 0f)));
        var planes = Rs(padded, 0, channels, hp * wp);
        long taps = kh * kw;
        long positions = ho * wo;
        var offsets = new long[taps * positions];
        for (long i = 0; i < kh; i++)
        {
            for (long j = 0; j < kw; j++)
            {
                for (long r = 0; r < ho; r++)
                {
                    for (long c = 0; c < wo; c++)
                    {
                        offsets[((i * kw + j) * ho + r) * wo + c] = (r * strides[0] + i * dilations[0]) * wp + c * strides[1] + j * dilations[1];
                    }
                }
            }
        }

        var indices = Idx(TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int32, [taps * positions], offsets, name + ".im2col")));
        var columns = Rs(Emit("GatherAxis", [planes, indices], Attributes.From(("axis", 2L))), 0, groups, perGroup * taps, positions);
        var weights = Rs(w0, 1, groups, outChannels / groups, perGroup * taps);
        var y = Emit("BatchMatMul", [weights, columns]);
        Heavy();
        y = Rs(y, 0, outChannels, ho, wo);
        if (ins.Length > 2 && ins[2] is not null)
        {
            y = Emit("Add", [y, Rs(AsValue(ins[2]!), outChannels, 1, 1)]);
        }

        SetDyn(n, 0, oneD ? Rs(y, 0, outChannels, wo) : y);
    }
}
