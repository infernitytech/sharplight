namespace Sharplight.Studio.Services.Runtime;

/// <summary>An encoded image (PNG, JPEG, ...) sent with a message.</summary>
/// <param name="Data">The file bytes.</param>
/// <param name="MediaType">MIME type.</param>
internal sealed record ImageContent(byte[] Data, string MediaType);
