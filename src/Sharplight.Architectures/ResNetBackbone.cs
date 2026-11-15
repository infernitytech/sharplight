using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// ResNet-family backbones over a feature map [1, C, H, W] with batch normalization folded into the convolutions (every convolution has a weight and a
/// bias): <c>{Prefix}stem.{i}.{weight,bias}</c> (one 7x7 or three 3x3 convolutions, then a 3x3 max pool), and per stage s and block b
/// <c>{Prefix}stages.{s}.{b}.conv{k}.{weight,bias}</c> plus <c>{Prefix}stages.{s}.{b}.short.{weight,bias}</c> where the identity does not fit.
/// </summary>
internal static class ResNetBackbone
{
    /// <summary>Builds the backbone and returns the output of every stage (strides 4, 8, 16, 32).</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="image">The input [1, 3, H, W].</param>
    /// <param name="o">The backbone shape.</param>
    public static Value[] Build(GraphBuilder b, Value image, ResNetOptions o)
    {
        string p = o.Prefix;
        Value x = image;
        if (o.DeepStem)
        {
            x = Conv(b, x, $"{p}stem.0", 2, 1, relu: true);
            x = Conv(b, x, $"{p}stem.1", 1, 1, relu: true);
            x = Conv(b, x, $"{p}stem.2", 1, 1, relu: true);
        }
        else
        {
            x = Conv(b, x, $"{p}stem.0", 2, 3, relu: true);
        }

        x = b.Emit1("Pool", [x], Attributes.From(("mode", "max"), ("kernel", new long[] { 3, 3 }), ("strides", new long[] { 2, 2 }), ("pads", new long[] { 1, 1, 1, 1 })));
        var outputs = new Value[o.Depths.Length];
        for (int s = 0; s < o.Depths.Length; s++)
        {
            for (int blk = 0; blk < o.Depths[s]; blk++)
            {
                int stride = blk == 0 && s > 0 ? 2 : 1;
                string name = $"{p}stages.{s}.{blk}.";
                var identity = x;
                Value y;
                if (o.Bottleneck)
                {
                    y = Conv(b, x, name + "conv1", o.StrideOnFirst1x1 ? stride : 1, 0, relu: true);
                    y = Conv(b, y, name + "conv2", o.StrideOnFirst1x1 ? 1 : stride, 1, relu: true);
                    y = Conv(b, y, name + "conv3", 1, 0, relu: false);
                }
                else
                {
                    y = Conv(b, x, name + "conv1", stride, 1, relu: true);
                    y = Conv(b, y, name + "conv2", 1, 1, relu: false);
                }

                if (b.HasWeight(name + "short.weight"))
                {
                    var shortcut = x;
                    if (stride == 2 && o.AvgDownsample)
                    {
                        shortcut = b.Emit1("Pool", [shortcut], Attributes.From(("mode", "avg"), ("kernel", new long[] { 2, 2 }), ("strides", new long[] { 2, 2 }), ("ceil_mode", true)));
                        identity = Conv(b, shortcut, name + "short", 1, 0, relu: false);
                    }
                    else
                    {
                        identity = Conv(b, shortcut, name + "short", stride, 0, relu: false);
                    }
                }

                x = b.Unary("Relu", b.Add(y, identity));
            }

            outputs[s] = x;
        }

        return outputs;
    }

    /// <summary>One convolution with folded bias and optional ReLU; <paramref name="pad"/> is the symmetric padding.</summary>
    public static Value Conv(GraphBuilder b, Value x, string name, int stride, int pad, bool relu)
    {
        var y = b.Emit1("Conv", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")],
            Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad })));
        return relu ? b.Unary("Relu", y) : y;
    }
}
