using System.Text.RegularExpressions;

namespace Sharplight.Gguf;

/// <summary>llama.cpp tensor naming shared by all architecture mappings.</summary>
internal static partial class GgufTensorNames
{
    /// <summary>Matches <c>blk.{layer}.{suffix}</c>; group 1 is the layer index, group 2 the per-layer tensor name.</summary>
    [GeneratedRegex(@"^blk\.(\d+)\.(.+)$")]
    public static partial Regex Block();

    /// <summary>Embedding / final norm / output head, which every decoder-style architecture names identically.</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["token_embd.weight"] = "token_embedding.weight",
        ["output_norm.weight"] = "output_norm.weight",
        ["output.weight"] = "lm_head.weight",
    };

    /// <summary>Canonical name of a global (non-layer) tensor, or null.</summary>
    public static string? MapGlobal(string ggufName) => s_global.GetValueOrDefault(ggufName);
}
