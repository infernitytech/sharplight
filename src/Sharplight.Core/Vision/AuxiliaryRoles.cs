namespace Sharplight.Vision;

/// <summary>Auxiliary source roles understood by adapters.</summary>
public static class AuxiliaryRoles
{
    /// <summary>A separate vision tower file (llama.cpp's "mmproj").</summary>
    public const string Mmproj = "mmproj";

    /// <summary>A speculative-decoding drafter trained for the primary model (an EAGLE-3 checkpoint); it becomes the model's draft graph.</summary>
    public const string Draft = "draft";
}
