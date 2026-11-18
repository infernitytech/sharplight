namespace Sharplight.Architectures;

/// <summary>
/// One OmniVoice sequence ready to unmask: the conditional input (the style tokens, the text tokens, the reference codes and the target positions' mask) as the language model sees it.
/// </summary>
/// <param name="Ids">Token rows [codebooks, <see cref="Length"/>] row-major; row 0 holds the text token ids, every row the audio codes of audio positions.</param>
/// <param name="AudioMask">Per position, whether it is an audio position (reference codes and targets) rather than a text one.</param>
/// <param name="TargetLength">The number of target frames (the last positions).</param>
/// <param name="ReferenceLength">The number of reference frames before the target.</param>
/// <param name="Codebooks">The number of codebook rows.</param>
internal sealed record OmniVoicePrompt(int[] Ids, bool[] AudioMask, int TargetLength, int ReferenceLength, int Codebooks)
{
    /// <summary>The number of positions.</summary>
    public int Length => AudioMask.Length;
}
