namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of an ECAPA-TDNN speaker encoder (<see cref="EcapaTdnnGraph"/>).</summary>
/// <param name="MelBins">The channels of the input log-mel frames.</param>
/// <param name="Embedding">The width of the speaker embedding.</param>
/// <param name="Channels">The channels of the initial TDNN layer, the SE-Res2Net blocks and the multi-layer aggregation (last).</param>
/// <param name="Kernels">The kernel size of each of those layers.</param>
/// <param name="Dilations">The dilation of each of those layers.</param>
/// <param name="AttentionChannels">The hidden channels of the attentive statistics pooling.</param>
/// <param name="Res2NetScale">The channel groups of a Res2Net block.</param>
/// <param name="SeChannels">The bottleneck channels of a squeeze-excitation block.</param>
internal sealed record EcapaTdnnParameters(int MelBins, int Embedding, int[] Channels, int[] Kernels, int[] Dilations, int AttentionChannels, int Res2NetScale, int SeChannels);
