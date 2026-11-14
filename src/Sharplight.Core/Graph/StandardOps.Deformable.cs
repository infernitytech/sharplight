using Sharplight.Tensors;

namespace Sharplight.Graph;

// Deformable convolution.
public static partial class StandardOps
{
    /// <summary>
    /// Deformable 2-D convolution (torchvision deform_conv2d / ONNX DeformConv): x [N, C, H, W], w [O, C/groups, kh, kw], offset
    /// [N, offset_groups·kh·kw·2, HO, WO] ((dy, dx) per kernel tap), bias [O] and optionally mask [N, offset_groups·kh·kw, HO, WO] →
    /// [N, O, HO, WO]. Every tap samples x bilinearly at its regular position plus the offset (zero outside), scaled by the mask.
    /// Attributes: <c>strides</c>, <c>pads</c> (top, left, bottom, right), <c>dilations</c>, <c>groups</c>, <c>offset_groups</c>.
    /// </summary>
    public static readonly OpSchema DeformConv = new("DeformConv", OpCategory.MatMul, 4, 5, (i, a) =>
    {
        var x = i[0].Shape;
        var w = i[1].Shape;
        var (ho, wo) = SpatialOut(x[2].Fixed, x[3].Fixed, w[2].Fixed, w[3].Fixed, a, ceil: false);
        return [new(DataType.F32, new Shape(x[0], w[0], ho, wo))];
    });
}
