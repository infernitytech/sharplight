using System.Text.RegularExpressions;

namespace Sharplight.Import.Gguf;

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

    /// <summary>
    /// Whether every block tensor of the file belongs to the multi-token-prediction (NextN) blocks that follow the trunk
    /// (<c>nextn_predict_layers</c>), as in the <c>mtp-*.gguf</c> draft files converters write next to the model.
    /// </summary>
    /// <param name="tensorNames">The GGUF tensor names of every shard.</param>
    /// <param name="gguf">The metadata view.</param>
    public static bool HoldsOnlyNextnBlocks(IEnumerable<string> tensorNames, GgufMetadataView gguf)
    {
        int nextn = gguf.Int("nextn_predict_layers", 0);
        if (nextn <= 0 || gguf.Raw("block_count") is null)
        {
            return false;
        }

        int trunk = gguf.Int("block_count") - nextn;
        bool any = false;
        foreach (string name in tensorNames)
        {
            var m = Block().Match(name);
            if (!m.Success)
            {
                continue;
            }

            if (int.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture) < trunk)
            {
                return false;
            }

            any = true;
        }

        return any;
    }
}
