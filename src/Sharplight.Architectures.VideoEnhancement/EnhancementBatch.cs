namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>One batch of a video-enhancement request: a run of input frames restored together.</summary>
/// <param name="Start">Index of its first input frame.</param>
/// <param name="Count">Input frames in the batch.</param>
/// <param name="PaddedFrames">Frames after padding with copies of the last frame (1 + k · the temporal compression).</param>
/// <param name="LatentFrames">Latent frames of the padded batch.</param>
internal sealed record EnhancementBatch(int Start, int Count, int PaddedFrames, int LatentFrames);
