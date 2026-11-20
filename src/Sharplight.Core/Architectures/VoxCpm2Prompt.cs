namespace Sharplight.Architectures;

/// <summary>
/// The rows of a VoxCPM2 prompt: per row a token (text rows) or a latent patch (audio rows), as the reference's <c>_generate_with_prompt_cache</c> lays them out
/// (an isolated reference <c>[reference start, patches, reference end]</c>, the text with the audio start token, then the continuation prompt's patches).
/// </summary>
/// <param name="Tokens">The token of every row (0 for audio rows).</param>
/// <param name="Audio">Whether a row is an audio row.</param>
/// <param name="Patches">The patches of the audio rows in row order ([patch, feature] row-major each).</param>
/// <param name="TargetTokens">The tokens of the text to speak (without the prompt text), which bound the generated length.</param>
/// <param name="ContinuationPatches">The trailing audio rows of the continuation prompt (0 without one).</param>
internal sealed record VoxCpm2Prompt(int[] Tokens, bool[] Audio, IReadOnlyList<float[]> Patches, int TargetTokens, int ContinuationPatches);
