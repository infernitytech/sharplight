namespace Sharplight.Studio.Models;

/// <summary>Which resident models the chat uses for voice and images, and whether answers are read aloud.</summary>
internal sealed record VoiceSettings
{
    /// <summary>The speech-recognition model that transcribes the microphone into the prompt; null uses the first resident one.</summary>
    public string? RecognitionModel { get; init; }

    /// <summary>The speech-synthesis model that reads answers aloud; null uses the first resident one.</summary>
    public string? SynthesisModel { get; init; }

    /// <summary>The image-generation model behind the <c>/image</c> command; null uses the first resident one.</summary>
    public string? ImageModel { get; init; }

    /// <summary>Read every finished answer aloud (needs a resident speech-synthesis model).</summary>
    public bool ReadAloud { get; init; }
}
