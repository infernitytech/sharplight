namespace Sharplight.Server;

/// <summary>Decodes base64 payloads of request bodies (images) and reports malformed input as an HTTP 400 error.</summary>
internal static class Base64Payload
{
    /// <summary>Decodes <paramref name="text"/> (null decodes as empty).</summary>
    /// <exception cref="ApiException">The text is not valid base64 (HTTP 400).</exception>
    public static byte[] Decode(string? text)
    {
        try
        {
            return Convert.FromBase64String(text ?? "");
        }
        catch (FormatException)
        {
            throw ApiException.BadRequest("Invalid base64 image data.");
        }
    }
}
