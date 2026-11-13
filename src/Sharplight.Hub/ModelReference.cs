using System.Diagnostics.CodeAnalysis;
using System.Text.RegularExpressions;

namespace Sharplight.Hub;

/// <summary>
/// A Hub model as named on the command line: <c>owner/name</c> with an optional <c>:tag</c> that selects a variant (a GGUF
/// quantization such as <c>Q4_K_M</c>, a file name, or an ONNX variant such as <c>fp16</c>). Links
/// (<c>https://huggingface.co/owner/name</c>, <c>hf.co/owner/name:tag</c>) are accepted too.
/// </summary>
/// <param name="Repository">The repository id, <c>owner/name</c>.</param>
/// <param name="Tag">The variant, or <see langword="null"/> for the default one.</param>
public sealed partial record ModelReference(string Repository, string? Tag)
{
    private const char TagSeparator = ':';
    private static readonly string[] s_prefixes = ["https://huggingface.co/", "http://huggingface.co/", "https://hf.co/", "hf.co/", "huggingface.co/"];

    /// <summary>Parses <paramref name="text"/>; returns false when it is not <c>owner/name[:tag]</c>.</summary>
    /// <param name="text">The text to parse.</param>
    /// <param name="reference">The parsed reference.</param>
    public static bool TryParse(string text, [NotNullWhen(true)] out ModelReference? reference)
    {
        reference = null;
        var trimmed = text.Trim();
        foreach (var prefix in s_prefixes.Where(prefix => trimmed.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)))
        {
            trimmed = trimmed[prefix.Length..];
            break;
        }

        int separator = trimmed.IndexOf(TagSeparator);
        var repository = separator >= 0 ? trimmed[..separator] : trimmed;
        var tag = separator >= 0 ? trimmed[(separator + 1)..] : null;
        if (!RepositoryPattern().IsMatch(repository) || tag is { Length: 0 })
        {
            return false;
        }

        reference = new ModelReference(repository, tag);
        return true;
    }

    /// <summary>Parses <paramref name="text"/>.</summary>
    /// <param name="text">The text to parse.</param>
    /// <exception cref="FormatException">The text is not <c>owner/name[:tag]</c>.</exception>
    public static ModelReference Parse(string text)
    {
        return TryParse(text, out var reference)
            ? reference
            : throw new FormatException($"'{text}' is not a Hugging Face model name; expected owner/name or owner/name:tag (e.g. Qwen/Qwen3-0.6B-GGUF:Q8_0).");
    }

    /// <summary><c>owner/name</c> or <c>owner/name:tag</c>.</summary>
    public override string ToString() => Tag is null ? Repository : $"{Repository}{TagSeparator}{Tag}";

    [GeneratedRegex(@"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")]
    private static partial Regex RepositoryPattern();
}
