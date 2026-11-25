namespace Sharplight.VideoEnhancement;

/// <summary>
/// What a video enhancer does without options and which sizes it takes (<see cref="IVideoEnhancer.Defaults"/>), for front
/// ends that offer the choices.
/// </summary>
/// <param name="Scale">The enlargement when the request gives no size.</param>
/// <param name="SizeMultiple">The network works on sizes that are multiples of this many pixels; other sizes are padded and cropped back.</param>
/// <param name="FrameMultiple">
/// Batches hold 1 + k · this many frames (the temporal compression of a causal video VAE); shorter batches are padded with copies of their last frame.
/// </param>
/// <param name="ColorCorrection">The default colour correction.</param>
/// <param name="FramesPerBatch">The default frames per batch, or 0 for the whole clip at once.</param>
public sealed record VideoEnhancementDefaults(float Scale, int SizeMultiple, int FrameMultiple, VideoColorCorrection ColorCorrection, int FramesPerBatch);
