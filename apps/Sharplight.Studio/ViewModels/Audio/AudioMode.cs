namespace Sharplight.Studio.ViewModels.Audio;

/// <summary>What the Audio page does with its input, decided by the kind of the chosen model.</summary>
internal enum AudioMode
{
    /// <summary>No model: the page shows the generic input.</summary>
    None,

    /// <summary>An audio codec: the input is encoded to codes and decoded back (a round trip).</summary>
    Codec,

    /// <summary>A voice activity detector: the speech segments of the input.</summary>
    VoiceActivity,
}
