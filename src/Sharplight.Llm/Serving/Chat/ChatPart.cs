namespace Sharplight.Llm.Serving.Chat;

/// <summary>A content part of a message: <c>text</c> or <c>image</c>.</summary>
public sealed record ChatPart(string Type, string? Text = null, ImageData? Image = null)
{
    public static ChatPart FromText(string text) => new("text", text);
    public static ChatPart FromImage(ImageData image) => new("image", null, image);
}
