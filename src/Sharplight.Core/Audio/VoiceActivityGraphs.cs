namespace Sharplight.Audio;

/// <summary>
/// The graph contract of the <see cref="Models.ModelKinds.VoiceActivity"/> kind: a component with the role <see cref="Models.ModelComponentRoles.Encoder"/> and the stateful
/// graph <see cref="StepGraph"/>, run once per frame: the frame's samples (with the model's context before them) in, the speech probability out, and the recurrent state in and out.
/// </summary>
public static class VoiceActivityGraphs
{
    /// <summary>The per-frame graph.</summary>
    public const string StepGraph = "vad_step";

    /// <summary>Input: f32 <c>[context + frame samples, 1]</c> (the last samples of the previous frames, then the frame).</summary>
    public const string AudioInput = "audio";

    /// <summary>Input: f32 recurrent state, all zeros at the start of an audio; the previous <see cref="StateOutput"/> afterwards.</summary>
    public const string StateInput = "state";

    /// <summary>Output: the new recurrent state (the shape of <see cref="StateInput"/>).</summary>
    public const string StateOutput = "state_out";

    /// <summary>Output: f32 <c>[1, 1]</c>, the speech probability of the frame.</summary>
    public const string ProbabilityOutput = "probability";

    /// <summary>The configuration key of the sample rate.</summary>
    public const string SampleRateKey = "sampleRate";

    /// <summary>The configuration key of the samples per frame.</summary>
    public const string FrameSamplesKey = "frameSamples";

    /// <summary>The configuration key of the context samples that precede a frame.</summary>
    public const string ContextSamplesKey = "contextSamples";
}
