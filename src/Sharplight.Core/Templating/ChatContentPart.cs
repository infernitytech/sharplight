namespace Sharplight.Templating;

/// <summary>One part of a multimodal message: <c>text</c>, <c>image</c> or <c>video</c>.</summary>
public sealed record ChatContentPart(string Type, string? Text = null)
{
    /// <summary>Creates a text part.</summary>
    public static ChatContentPart FromText(string text) => new("text", text);
    /// <summary>Creates an image placeholder part; the template expands it to the model's image tokens.</summary>
    public static ChatContentPart Image() => new("image");
    /// <summary>Creates a video placeholder part.</summary>
    public static ChatContentPart Video() => new("video");
}
