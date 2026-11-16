using System.Globalization;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Maps Z-Image transformer tensors to the canonical <c>transformer.*</c> vocabulary of the <c>zimage</c> architecture (documented on
/// <c>ZImageTransformerGraphBuilder</c>), from diffusers' <c>ZImageTransformer2DModel</c> names (separate q/k/v joined lazily) and from the
/// original names of the single files and GGUF files (fused <c>attention.qkv</c>, <c>attention.out</c>, <c>q_norm</c> / <c>k_norm</c>,
/// <c>x_embedder</c> / <c>final_layer</c> without the patch-size key).
/// </summary>
internal static partial class ZImageNames
{
    private const string Canonical = "transformer.";

    /// <summary>Whether the names are a Z-Image transformer in the original layout (after removing <paramref name="prefix"/>).</summary>
    /// <param name="names">Tensor names.</param>
    /// <param name="prefix">Prefix in front of the transformer names.</param>
    public static bool IsOriginal(IEnumerable<string> names, string prefix)
        => names.Any(n => n == prefix + "cap_pad_token") && names.Any(n => n == prefix + "noise_refiner.0.attention.qkv.weight" || n == prefix + "noise_refiner.0.attention.to_q.weight");

    /// <summary>Adds transformer tensors of either layout (names after <paramref name="prefix"/>; other tensors are ignored).</summary>
    /// <param name="tensors">The tensors by stored name.</param>
    /// <param name="prefix">Prefix in front of the transformer names.</param>
    /// <param name="weights">The canonical set.</param>
    /// <exception cref="InvalidDataException">A transformer tensor has no canonical name.</exception>
    public static void Add(IReadOnlyDictionary<string, WeightInfo> tensors, string prefix, CanonicalWeights weights)
    {
        var fused = new Dictionary<string, SortedDictionary<int, WeightInfo>>(StringComparer.Ordinal);
        foreach (var (stored, weight) in tensors)
        {
            if (!stored.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }

            var (canonical, part) = CanonicalName(stored[prefix.Length..]);
            if (canonical is null)
            {
                throw new InvalidDataException($"Unexpected Z-Image transformer tensor '{stored}'.");
            }

            if (part < 0)
            {
                weights.Add(Canonical + canonical, weight);
                continue;
            }

            if (!fused.TryGetValue(canonical, out var parts))
            {
                fused[canonical] = parts = [];
            }

            parts[part] = weight;
        }

        foreach (var (canonical, parts) in fused)
        {
            weights.Add(Canonical + canonical, WeightInfo.ConcatRows(Canonical + canonical, [.. parts.Values]));
        }
    }

    /// <summary>The number of blocks of a stack (<c>layers</c>, <c>noise_refiner</c>, <c>context_refiner</c>) in canonical weights.</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <param name="stack">The stack.</param>
    public static int CountBlocks(IReadOnlyDictionary<string, WeightInfo> weights, string stack)
    {
        int count = 0;
        while (weights.ContainsKey(string.Create(CultureInfo.InvariantCulture, $"{Canonical}{stack}.{count}.attention.out.weight")))
        {
            count++;
        }

        return count;
    }

    /// <summary>The canonical name of a tensor of either layout, and its row block when several are joined into it (else −1).</summary>
    private static (string? Canonical, int Part) CanonicalName(string name)
    {
        switch (name)
        {
            case "x_pad_token" or "cap_pad_token":
                return (name, -1);
            case "cap_embedder.0.weight":
                return ("cap_embedder.norm.weight", -1);
        }

        var top = TopLevel().Match(name);
        if (top.Success)
        {
            string suffix = top.Groups[2].Value;
            string? module = top.Groups[1].Value switch
            {
                "x_embedder" or "all_x_embedder.2-1" => "x_embedder",
                "final_layer.linear" or "all_final_layer.2-1.linear" => "final.linear",
                "final_layer.adaLN_modulation.1" or "all_final_layer.2-1.adaLN_modulation.1" => "final.mod",
                "t_embedder.mlp.0" => "t_embedder.in_layer",
                "t_embedder.mlp.2" => "t_embedder.out_layer",
                "cap_embedder.1" => "cap_embedder.linear",
                _ => null,
            };
            if (module is not null)
            {
                return ($"{module}.{suffix}", -1);
            }
        }

        var block = Block().Match(name);
        if (!block.Success)
        {
            return (null, -1);
        }

        string p = $"{block.Groups[1].Value}.{block.Groups[2].Value}.";
        string rest = block.Groups[3].Value;
        return rest switch
        {
            "attention.qkv.weight" => (p + "attention.qkv.weight", -1),
            "attention.to_q.weight" => (p + "attention.qkv.weight", 0),
            "attention.to_k.weight" => (p + "attention.qkv.weight", 1),
            "attention.to_v.weight" => (p + "attention.qkv.weight", 2),
            "attention.out.weight" or "attention.to_out.0.weight" => (p + "attention.out.weight", -1),
            "attention.q_norm.weight" or "attention.norm_q.weight" => (p + "attention.norm_q.weight", -1),
            "attention.k_norm.weight" or "attention.norm_k.weight" => (p + "attention.norm_k.weight", -1),
            "attention_norm1.weight" or "attention_norm2.weight" or "ffn_norm1.weight" or "ffn_norm2.weight" => (p + rest, -1),
            "feed_forward.w1.weight" => (p + "ffn.w1.weight", -1),
            "feed_forward.w2.weight" => (p + "ffn.w2.weight", -1),
            "feed_forward.w3.weight" => (p + "ffn.w3.weight", -1),
            "adaLN_modulation.0.weight" => (p + "mod.weight", -1),
            "adaLN_modulation.0.bias" => (p + "mod.bias", -1),
            _ => (null, -1),
        };
    }

    [GeneratedRegex(@"^(.+)\.(weight|bias)$")]
    private static partial Regex TopLevel();

    [GeneratedRegex(@"^(layers|noise_refiner|context_refiner)\.(\d+)\.(.+)$")]
    private static partial Regex Block();
}
