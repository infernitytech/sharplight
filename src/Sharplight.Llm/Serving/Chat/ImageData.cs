namespace Sharplight.Llm.Serving.Chat;

/// <summary>An image attached to a message: encoded file bytes (PNG/JPEG/...) or a URL.</summary>
public sealed record ImageData(byte[]? Bytes, string? MediaType, string? Url);
