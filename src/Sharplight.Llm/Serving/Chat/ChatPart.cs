namespace Sharplight.Llm.Serving.Chat;

/// <summary>A content part of a message: <c>text</c> or <c>image</c>.</summary>
/// <param name="Type"><c>text</c> or <c>image</c>.</param>
/// <param name="Text">The text of a <c>text</c> part.</param>
/// <param name="Image">The picture of an <c>image</c> part.</param>
public sealed record ChatPart(string Type, string? Text = null, ImageData? Image = null)
{
    /// <summary>Creates a text part.</summary>
    public static ChatPart FromText(string text) => new("text", text);
    /// <summary>Creates an image part.</summary>
    public static ChatPart FromImage(ImageData image) => new("image", null, image);
}
