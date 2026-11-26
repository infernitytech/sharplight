namespace Sharplight.Architectures;

/// <summary>The talker's input of a Qwen3-TTS request (<see cref="Qwen3TtsSynthesizer.BuildPrompt"/>).</summary>
/// <param name="Rows">The prompt rows [rows, hidden], row-major.</param>
/// <param name="Trailing">The text rows [steps, hidden] added to the inputs of the first generation steps (interleaved text); after them the text pad row is added.</param>
/// <param name="ReferenceFrames">The reference recording's frames the generated ones continue (decoded with them as context), or empty.</param>
internal sealed record Qwen3TtsPrompt(float[] Rows, float[] Trailing, IReadOnlyList<int[]> ReferenceFrames);
