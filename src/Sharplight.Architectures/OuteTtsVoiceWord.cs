namespace Sharplight.Architectures;

/// <summary>One word of an OuteTTS speaker profile: the word, how long it is spoken and the codec tokens of its audio.</summary>
/// <param name="Word">The (normalized) word.</param>
/// <param name="Duration">The duration in seconds.</param>
/// <param name="Codes">The codec tokens of the spoken word.</param>
internal sealed record OuteTtsVoiceWord(string Word, double Duration, int[] Codes);
