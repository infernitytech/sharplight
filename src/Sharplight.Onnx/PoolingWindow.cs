namespace Sharplight.Onnx;

/// <summary>Window of an N-D pooling node: per spatial axis kernel, stride and dilation, plus explicit pads [begins..., ends...].</summary>
/// <param name="Kernel">Window extent per spatial axis.</param>
/// <param name="Strides">Stride per spatial axis.</param>
/// <param name="Dilations">Dilation per spatial axis.</param>
/// <param name="Pads">Begin pads of every spatial axis followed by the end pads.</param>
internal sealed record PoolingWindow(long[] Kernel, long[] Strides, long[] Dilations, long[] Pads);
