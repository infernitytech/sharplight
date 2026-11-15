namespace Sharplight.Llm.Serving.Chat;

/// <summary>An image attached to a message: encoded file bytes (PNG/JPEG/...) or a URL.</summary>
/// <param name="Bytes">Encoded image file contents, when supplied inline.</param>
/// <param name="MediaType">MIME type, for example <c>image/png</c>.</param>
/// <param name="Url">Where the image can be fetched, when it is not inline.</param>
public sealed record ImageData(byte[]? Bytes, string? MediaType, string? Url);
