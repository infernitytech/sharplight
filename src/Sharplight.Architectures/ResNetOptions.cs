namespace Sharplight.Architectures;

/// <summary>Shape of a ResNet-family convolutional backbone (<see cref="ResNetBackbone"/>).</summary>
/// <param name="Prefix">Weight name prefix.</param>
/// <param name="Depths">Blocks per stage.</param>
/// <param name="Bottleneck">True for bottleneck blocks (1x1, 3x3, 1x1 with 4x expansion), false for basic blocks (two 3x3).</param>
/// <param name="DeepStem">True for the three 3x3 convolutions stem of ResNet-D, false for one 7x7.</param>
/// <param name="AvgDownsample">True for ResNet-D shortcuts (2x2 average pooling before the 1x1 convolution of a strided stage), false for a strided 1x1 convolution.</param>
/// <param name="StrideOnFirst1x1">True when a bottleneck strides in its first 1x1 convolution (the original ResNet-B variant strides in the 3x3).</param>
internal sealed record ResNetOptions(string Prefix, int[] Depths, bool Bottleneck, bool DeepStem, bool AvgDownsample, bool StrideOnFirst1x1 = false);
