namespace Sharplight.Architectures;

/// <summary>
/// A Chatterbox voice: what T3 and S3Gen condition on (the built-in <c>conds.pt</c>, or a recording through <see cref="ChatterboxVoiceCloning"/>).
/// </summary>
/// <param name="Speaker">The voice encoder's speaker embedding [256].</param>
/// <param name="PromptTokens">T3's prompt speech tokens.</param>
/// <param name="FlowTokens">The S3 tokens of the flow prompt.</param>
/// <param name="FlowMel">The flow prompt's log-mel [2 · flow tokens, 80] (time-major).</param>
/// <param name="FlowSpeaker">The CAM++ x-vector [192].</param>
internal sealed record ChatterboxVoice(float[] Speaker, int[] PromptTokens, int[] FlowTokens, float[] FlowMel, float[] FlowSpeaker);
