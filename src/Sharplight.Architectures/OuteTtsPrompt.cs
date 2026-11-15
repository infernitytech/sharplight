using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The prompt format of OuteTTS 0.2: the text is normalized to lowercase words (numbers spelled out, punctuation dropped), joined by
/// <c>&lt;|text_sep|&gt;</c> inside <c>&lt;|text_start|&gt;</c> / <c>&lt;|text_end|&gt;</c>, followed by <c>&lt;|audio_start|&gt;</c>; the model answers
/// with <c>word&lt;|t_0.64|&gt;&lt;|code_start|&gt;&lt;|1007|&gt;…&lt;|code_end|&gt;</c> groups. A speaker profile puts its words before the text and its
/// audio groups after <c>&lt;|audio_start|&gt;</c>, so the model continues in that voice.
/// </summary>
internal static partial class OuteTtsPrompt
{
    /// <summary>The end of the generated audio.</summary>
    public const string AudioEnd = "<|audio_end|>";

    /// <summary>The model version a language-model source is, from its name (<c>OuteTTS 0.2 500M</c> gives <c>0.2</c>), or <see langword="null"/>.</summary>
    /// <param name="source">The language-model source.</param>
    public static string? VersionOf(ModelSource source)
    {
        var match = VersionName().Match(source.Get(CanonicalKeys.Name, ""));
        return match.Success && match.Groups[1].Value == "0.2" ? match.Groups[1].Value : null;
    }

    /// <summary>The normalized words of a text: lowercase, numbers spelled out, only letters.</summary>
    /// <param name="text">The text to speak.</param>
    public static IReadOnlyList<string> Words(string text)
    {
        text = Number().Replace(text.ToLowerInvariant(), m => EnglishNumberWords.Spell(m.Value));
        text = Separators().Replace(text, " ");
        text = NonLetters().Replace(text, "");
        return Spaces().Replace(text, " ").Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    /// <summary>The prompt that makes the model speak <paramref name="words"/>, optionally in the voice of <paramref name="voice"/>.</summary>
    /// <param name="words">The normalized words to speak.</param>
    /// <param name="voice">The speaker profile, or <see langword="null"/> for the model's own voice.</param>
    public static string Build(IReadOnlyList<string> words, OuteTtsVoice? voice)
    {
        var prompt = new StringBuilder("<|im_start|>\n<|text_start|>");
        if (voice is not null)
        {
            foreach (var spoken in voice.Words)
            {
                prompt.Append(spoken.Word).Append("<|text_sep|>");
            }
        }

        prompt.Append(string.Join("<|text_sep|>", words)).Append("<|text_end|>\n<|audio_start|>\n");
        if (voice is not null)
        {
            foreach (var spoken in voice.Words)
            {
                prompt.Append(spoken.Word).Append(string.Create(CultureInfo.InvariantCulture, $"<|t_{spoken.Duration:F2}|><|code_start|>"));
                foreach (int code in spoken.Codes)
                {
                    prompt.Append("<|").Append(code.ToString(CultureInfo.InvariantCulture)).Append("|>");
                }

                prompt.Append("<|code_end|>\n");
            }
        }

        return prompt.ToString();
    }

    /// <summary>The codec token a generated token stands for (<c>&lt;|1007|&gt;</c> is 1007), or -1 for any other token.</summary>
    /// <param name="text">The token's text.</param>
    public static int CodeOf(string text)
        => CodeToken().Match(text) is { Success: true } match ? int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture) : -1;

    [GeneratedRegex(@"OuteTTS\s+(\d+\.\d+)", RegexOptions.IgnoreCase)]
    private static partial Regex VersionName();

    [GeneratedRegex(@"\d+(\.\d+)?")]
    private static partial Regex Number();

    [GeneratedRegex(@"[-_/,\.\\]")]
    private static partial Regex Separators();

    [GeneratedRegex(@"[^a-z\s]")]
    private static partial Regex NonLetters();

    [GeneratedRegex(@"\s+")]
    private static partial Regex Spaces();

    [GeneratedRegex(@"^<\|(\d+)\|>$")]
    private static partial Regex CodeToken();
}
