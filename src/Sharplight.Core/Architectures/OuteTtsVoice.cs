using System.Text.Json;

namespace Sharplight.Architectures;

/// <summary>
/// An OuteTTS speaker profile (the stored form of a voice, <c>voices/{name}.json</c> attachment): a short reference utterance as words with
/// durations and codec tokens, which the prompt puts before the text to speak so the model continues in that voice.
/// </summary>
/// <param name="Words">The words of the reference utterance.</param>
internal sealed record OuteTtsVoice(OuteTtsVoiceWord[] Words)
{
    /// <summary>Reads a profile from its JSON form.</summary>
    /// <param name="json">The JSON bytes.</param>
    public static OuteTtsVoice Parse(ReadOnlySpan<byte> json)
        => JsonSerializer.Deserialize(json, OuteTtsJsonContext.Default.OuteTtsVoice) ?? throw new InvalidDataException("Empty OuteTTS speaker profile.");

    /// <summary>The JSON form of the profile.</summary>
    public byte[] ToJson() => JsonSerializer.SerializeToUtf8Bytes(this, OuteTtsJsonContext.Default.OuteTtsVoice);
}
