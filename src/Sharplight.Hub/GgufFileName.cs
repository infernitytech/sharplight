using System.Globalization;
using System.Text.RegularExpressions;

namespace Sharplight.Hub;

/// <summary>
/// Reads what GGUF file names on the Hub encode: the quantization label (<c>Qwen3-0.6B-Q4_K_M.gguf</c> → <c>Q4_K_M</c>) and the
/// shard of a split model (<c>name-00002-of-00003.gguf</c>). Only the file name is used, never the file contents.
/// </summary>
internal static partial class GgufFileName
{
    public const string Extension = ".gguf";
    private const string ProjectorMarker = "mmproj";

    public static bool IsGguf(string path) => path.EndsWith(Extension, StringComparison.OrdinalIgnoreCase);

    /// <summary>A vision projector (llama.cpp's <c>mmproj</c>) rather than a language model.</summary>
    public static bool IsProjector(string path) => path.Contains(ProjectorMarker, StringComparison.OrdinalIgnoreCase);

    /// <summary>The path without <c>.gguf</c> and without the split suffix: the same for every shard of one model.</summary>
    public static string ModelKey(string path)
    {
        var stem = path[..^Extension.Length];
        var split = SplitSuffix().Match(stem);
        return split.Success ? stem[..split.Index] : stem;
    }

    /// <summary>The shard index and count of a split file (1-based), or null for an unsplit file.</summary>
    public static (int Index, int Count)? Shard(string path)
    {
        var split = SplitSuffix().Match(path[..^Extension.Length]);
        if (!split.Success)
        {
            return null;
        }

        int index = int.Parse(split.Groups["index"].Value, CultureInfo.InvariantCulture);
        int count = int.Parse(split.Groups["count"].Value, CultureInfo.InvariantCulture);
        return (index, count);
    }

    /// <summary>The upper-case quantization label at the end of the file name (<c>Q4_K_M</c>, <c>UD-Q4_K_XL</c>, <c>BF16</c>), or null.</summary>
    public static string? QuantLabel(string path)
    {
        var key = ModelKey(path);
        var match = Quantization().Match(RepoPath.Name(key));
        return match.Success ? match.Groups["label"].Value.ToUpperInvariant() : null;
    }

    [GeneratedRegex(@"-(?<index>\d{5})-of-(?<count>\d{5})$")]
    private static partial Regex SplitSuffix();

    [GeneratedRegex(@"(?:^|[-_.])(?<label>(?:UD-)?(?:I?Q[1-8](?:_[A-Z0-9]+)*|TQ[12]_0|MXFP4(?:_MOE)?|NVFP4|BF16|F16|F32|FP16|FP32))$", RegexOptions.IgnoreCase)]
    private static partial Regex Quantization();
}
