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
