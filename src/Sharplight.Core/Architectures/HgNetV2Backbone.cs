using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>The stage tables of an HGNetV2 backbone (PP-HGNetV2 L by default).</summary>
/// <param name="StemChannels">Stem channels (input, first, last).</param>
/// <param name="InChannels">Input channels per stage.</param>
/// <param name="MidChannels">Channels of the layers inside a block, per stage.</param>
/// <param name="OutChannels">Output channels per stage.</param>
/// <param name="Blocks">Blocks per stage.</param>
/// <param name="Layers">Layers per block, per stage.</param>
/// <param name="Downsample">Whether a stage starts with a depthwise stride-2 convolution.</param>
/// <param name="Light">Whether a stage uses light layers (1x1 then depthwise).</param>
/// <param name="Kernels">Layer kernel size per stage.</param>
internal sealed record HgNetV2Options(int[] StemChannels, int[] InChannels, int[] MidChannels, int[] OutChannels, int[] Blocks, int[] Layers, bool[] Downsample, bool[] Light, int[] Kernels);

/// <summary>
/// HGNetV2 backbones over [1, C, H, W] with batch normalization folded into every convolution (<c>.weight</c> and <c>.bias</c>): the stem
/// <c>{prefix}stem.{stem1,stem2a,stem2b,stem3,stem4}</c> (2x2 convolutions on bottom/right padded maps, a 2x2 stride-1 max pool of the first, channel concatenation),
/// and per stage s, block b: <c>{prefix}stages.{s}.down</c> (depthwise stride 2, no activation), <c>{prefix}stages.{s}.{b}.layers.{l}</c> (a convolution, or for light layers
/// <c>.conv1</c> 1x1 without activation and <c>.conv2</c> depthwise) and <c>{prefix}stages.{s}.{b}.agg.{0,1}</c> (1x1 squeeze and excitation of the concatenated layer outputs,
/// residual from the second block on). ReLU everywhere the reference has an activation.
/// </summary>
internal static class HgNetV2Backbone
{
    /// <summary>Builds the backbone and returns the output of every stage (strides 4, 8, 16, 32 for the default tables).</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="image">The input [1, 3, H, W].</param>
    /// <param name="prefix">Weight name prefix.</param>
    /// <param name="o">The stage tables.</param>
    public static Value[] Build(GraphBuilder b, Value image, string prefix, HgNetV2Options o)
    {
        var embedding = Conv(b, image, prefix + "stem.stem1", 2, 3, 1, true);
        embedding = PadEnd(b, embedding);
        var half = Conv(b, embedding, prefix + "stem.stem2a", 1, 2, 1, true);
        half = Conv(b, PadEnd(b, half), prefix + "stem.stem2b", 1, 2, 1, true);
        var pooled = b.Emit1("Pool", [embedding], Attributes.From(("mode", "max"), ("kernel", new long[] { 2, 2 }), ("strides", new long[] { 1, 1 })));
        var x = Conv(b, b.Concat(1, pooled, half), prefix + "stem.stem3", 2, 3, 1, true);
        x = Conv(b, x, prefix + "stem.stem4", 1, 1, 1, true);
        var outputs = new Value[o.InChannels.Length];
        for (int s = 0; s < outputs.Length; s++)
        {
            string stage = $"{prefix}stages.{s}.";
            if (o.Downsample[s])
            {
                x = Conv(b, x, stage + "down", 2, 3, o.InChannels[s], false);
            }

            for (int blk = 0; blk < o.Blocks[s]; blk++)
            {
                x = Block(b, x, $"{stage}{blk}.", o, s, blk);
            }

            outputs[s] = x;
        }

        return outputs;
    }

    private static Value Block(GraphBuilder b, Value x, string name, HgNetV2Options o, int stage, int block)
    {
        var pieces = new List<Value> { x };
        var y = x;
        for (int l = 0; l < o.Layers[stage]; l++)
        {
            string layer = $"{name}layers.{l}";
            if (o.Light[stage])
            {
                y = Conv(b, y, layer + ".conv1", 1, 1, 1, false);
                y = Conv(b, y, layer + ".conv2", 1, o.Kernels[stage], o.MidChannels[stage], true);
            }
            else
            {
                y = Conv(b, y, layer, 1, o.Kernels[stage], 1, true);
            }

            pieces.Add(y);
        }

        var aggregated = Conv(b, b.Concat(1, [.. pieces]), name + "agg.0", 1, 1, 1, true);
        aggregated = Conv(b, aggregated, name + "agg.1", 1, 1, 1, true);
        return block > 0 ? b.Add(aggregated, x) : aggregated;
    }

    // Zero padding of one row at the bottom and one column at the right.
    private static Value PadEnd(GraphBuilder b, Value x)
        => b.Emit1("Pad", [x], Attributes.From(("pads", new long[] { 0, 0, 0, 0, 0, 0, 1, 1 }), ("mode", "constant"), ("value", 0f)));

    // A convolution with a folded bias; the padding is (kernel - 1) / 2 as in the reference.
    private static Value Conv(GraphBuilder b, Value x, string name, int stride, int kernel, int groups, bool relu)
    {
        int pad = (kernel - 1) / 2;
        var y = b.Emit1("Conv", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")],
            Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad }), ("groups", (long)groups)));
        return relu ? b.Unary("Relu", y) : y;
    }
}
