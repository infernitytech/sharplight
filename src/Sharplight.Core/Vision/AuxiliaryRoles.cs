namespace Sharplight.Vision;

/// <summary>Auxiliary source roles understood by adapters.</summary>
public static class AuxiliaryRoles
{
    /// <summary>A separate vision tower file (llama.cpp's "mmproj").</summary>
    public const string Mmproj = "mmproj";

    /// <summary>A speculative-decoding drafter trained for the primary model (an EAGLE-3 checkpoint); it becomes the model's draft graph.</summary>
    public const string Draft = "draft";

    /// <summary>The decoder of an audio codec that turns the codec tokens a speech language model generates into a waveform (OuteTTS's WavTokenizer).</summary>
    public const string Vocoder = "vocoder";
}
