namespace Sharplight.Llm.Serving;

/// <summary>
/// A decoder that reports the time its last step stands for instead of the wall-clock time the engine would measure (test decoders
/// that simulate a device: their prices must not depend on how busy the host is).
/// </summary>
internal interface ISimulatedStepTime
{
    /// <summary>The time of the most recent step, or null to let the engine measure it.</summary>
    public TimeSpan? LastStepTime { get; }
}
