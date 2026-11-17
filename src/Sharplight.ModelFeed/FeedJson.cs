using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace Sharplight.ModelFeed;

/// <summary>Reads and writes feed documents with <see cref="FeedJsonContext"/>; the bytes written are the bytes that are hashed and signed.</summary>
public static class FeedJson
{
    /// <summary>Deserializes a document.</summary>
    /// <typeparam name="T">Document type.</typeparam>
    /// <param name="bytes">UTF-8 JSON.</param>
    /// <param name="type">Type metadata from <see cref="FeedJsonContext.Default"/>.</param>
    /// <param name="what">The document, for the message.</param>
    /// <returns>The document.</returns>
    /// <exception cref="FeedFormatException">The JSON is malformed or empty.</exception>
    public static T Read<T>(ReadOnlySpan<byte> bytes, JsonTypeInfo<T> type, string what)
    {
        try
        {
            return JsonSerializer.Deserialize(bytes, type) ?? throw new FeedFormatException($"{what} is empty.");
        }
        catch (JsonException exception)
        {
            throw new FeedFormatException($"{what} is not valid: {exception.Message}", exception);
        }
    }

    /// <summary>Serializes a document as indented UTF-8 JSON with a trailing newline.</summary>
    /// <typeparam name="T">Document type.</typeparam>
    /// <param name="value">The document.</param>
    /// <param name="type">Type metadata from <see cref="FeedJsonContext.Default"/>.</param>
    /// <returns>The bytes.</returns>
    public static byte[] Write<T>(T value, JsonTypeInfo<T> type)
    {
        var json = JsonSerializer.SerializeToUtf8Bytes(value, type);
        var bytes = new byte[json.Length + 1];
        json.CopyTo(bytes, 0);
        bytes[^1] = (byte)'\n';
        return bytes;
    }

    /// <summary>Parses and validates a manifest (schema version, id, version, package file).</summary>
    /// <param name="bytes">Manifest bytes.</param>
    /// <returns>The manifest.</returns>
    /// <exception cref="FeedFormatException">The manifest is malformed.</exception>
    public static PackageManifest ReadManifest(ReadOnlySpan<byte> bytes)
    {
        var manifest = Read(bytes, FeedJsonContext.Default.PackageManifest, "The manifest");
        if (manifest.SchemaVersion != 1)
        {
            throw new FeedFormatException($"Manifest schema version {manifest.SchemaVersion} is not supported (1 is).");
        }

        if (!FeedLayout.IsValidId(manifest.Id))
        {
            throw new FeedFormatException($"'{manifest.Id}' is not a valid package id (lower case letters, digits, dots and hyphens).");
        }

        if (!Versioning.SemanticVersion.TryParse(manifest.Version, out _))
        {
            throw new FeedFormatException($"'{manifest.Version}' is not a semantic version.");
        }

        return manifest;
    }
}
