namespace Sharplight.Architectures;

/// <summary>One turn of a VibeVoice script.</summary>
/// <param name="Speaker">The speaker's label as written in the script (<c>1</c> for <c>Speaker 1:</c>).</param>
/// <param name="Text">What the speaker says.</param>
internal sealed record VibeVoiceTurn(string Speaker, string Text);
