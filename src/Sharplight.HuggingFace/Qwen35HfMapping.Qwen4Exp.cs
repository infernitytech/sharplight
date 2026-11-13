using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Qwen3.8-Flash-Next (HF <c>qwen4_exp</c>): hyper-connections instead of the layer norms, the hashed n-gram embedding (PLE) on
/// one layer, the QSA block indexer on the attention layers and a sigmoid linear-attention output gate.
/// </summary>
internal sealed partial class Qwen35HfMapping
{
    private const string Qwen4ExpModelType = "qwen4_exp";
    private const string Qwen4ExpTextModelType = "qwen4_exp_text";

    // splitmix64 constants and the layer seed stride of the reference n-gram hash.
    private const ulong SplitMixGamma = 0x9E3779B97F4A7C15;
    private const ulong SplitMixM1 = 0xBF58476D1CE4E5B9;
    private const ulong SplitMixM2 = 0x94D049BB133111EB;
    private const ulong LayerSeedStride = 10007;

    [GeneratedRegex(@"^ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")]
    private static partial Regex NGramShardName();

    /// <summary>Hyper-connection mixer tensors, HF suffix to canonical suffix (the norm folds the +1).</summary>
    private static readonly Dictionary<string, string> s_hyperConnectionPlain = new(StringComparer.Ordinal)
    {
        ["input_mix_weight_down.weight"] = "down.weight",
        ["input_mix_weight_up.weight"] = "up.weight",
        ["block_inject_weight.weight"] = "inject.weight",
    };

    /// <summary>Per-layer qwen4_exp tensors that are only renamed or get the +1 norm fold.</summary>
    private static readonly Dictionary<string, (string Name, bool NormPlusOne)> s_qwen4ExpLayer = new(StringComparer.Ordinal)
    {
        ["ple.key_proj.weight"] = ("ngram.key.weight", false),
        ["ple.value_proj.weight"] = ("ngram.value.weight", false),
        ["ple.norm_key.weight"] = ("ngram.key_norm.weight", true),
        ["ple.norm_query.weight"] = ("ngram.query_norm.weight", true),
        ["ple.norm_conv.weight"] = ("ngram.conv_norm.weight", true),
        ["self_attn.indexer.q_layernorm.weight"] = ("attention.indexer.q_norm.weight", true),
        ["self_attn.indexer.k_layernorm.weight"] = ("attention.indexer.k_norm.weight", true),
    };

    private static bool IsQwen4Exp(HfConfig c) => c.ModelType is Qwen4ExpModelType or Qwen4ExpTextModelType;

    /// <summary>Hyper-connection, n-gram embedding, indexer and output-gate hyper-parameters.</summary>
    /// <exception cref="NotSupportedException">More than one n-gram embedding layer.</exception>
    private static void AddQwen4ExpHyperParameters(HfConfig c, Dictionary<string, object> md, string a)
    {
        md[a + "hyper_connections.count"] = c.Int("hc_count");
        md[a + "linear_attention.gate_activation"] = c.GetString("output_gate_type") ?? "silu";
        md[a + "attention.indexer.head_count"] = c.Int("indexer_n_heads");
        md[a + "attention.indexer.head_dim"] = c.Int("indexer_head_dim");
        md[a + "attention.indexer.block_size"] = c.Int("indexer_compress_ratio");
        md[a + "attention.indexer.token_budget"] = c.Int("indexer_budget");
        if ((c.GetInt("indexer_kv_heads") ?? 1) != 1)
        {
            throw new NotSupportedException("qwen4_exp: only one indexer key head is supported.");
        }

        var pleLayers = c.Get("ple_layer_ids") is { ValueKind: System.Text.Json.JsonValueKind.Array } ids
            ? ids.EnumerateArray().Select(x => x.GetInt32()).ToArray() : [];
        if (pleLayers.Length == 0)
        {
            return;
        }

        if (pleLayers.Length > 1)
        {
            throw new NotSupportedException("qwen4_exp: only one n-gram embedding (PLE) layer is supported.");
        }

        // The reference derives the hash constants from the config (they are also stored as checkpoint buffers).
        int order = c.Int("ngram_size");
        int headsPerOrder = c.Int("heads_per_ngram");
        const int pleIndex = 0;
        var sizes = HeadSizes(c.Int("ngram_vocab_size_base"), (order - 1) * headsPerOrder, pleIndex);
        var offsets = new long[sizes.Length];
        for (int h = 1; h < sizes.Length; h++)
        {
            offsets[h] = offsets[h - 1] + sizes[h - 1];
        }

        md[a + "ngram_embedding.layer"] = pleLayers[0] - 1;   // 1-based in the config
        md[a + "ngram_embedding.multipliers"] = Multipliers(c.Int("vocab_size"), order, pleIndex, (ulong)(c.GetInt("seed") ?? 1234));
        md[a + "ngram_embedding.head_sizes"] = sizes;
        md[a + "ngram_embedding.head_offsets"] = offsets;
        md[a + "ngram_embedding.heads_per_order"] = headsPerOrder;
        md[a + "ngram_embedding.stop_token"] = StopToken(c);
        md[a + "ngram_embedding.conv_dilation"] = order;
    }

    private static long StopToken(HfConfig c) => c.Get("eos_token_id") switch
    {
        { ValueKind: System.Text.Json.JsonValueKind.Array } list => list[0].GetInt64(),
        { } single => single.GetInt64(),
        null => throw new InvalidDataException("qwen4_exp: eos_token_id is required (it resets the n-gram context)."),
    };

    private static ulong SplitMix64(ulong value)
    {
        value += SplitMixGamma;
        value = (value ^ (value >> 30)) * SplitMixM1;
        value = (value ^ (value >> 27)) * SplitMixM2;
        return value ^ (value >> 31);
    }

    /// <summary>Odd multipliers small enough that token · multiplier never overflows a signed 64-bit integer.</summary>
    private static long[] Multipliers(int vocab, int order, int pleIndex, ulong seed)
    {
        ulong halfBound = Math.Max(1, (ulong)(long.MaxValue / Math.Max(vocab, 1)) / 2);
        ulong baseSeed = seed + LayerSeedStride * (ulong)pleIndex;
        return [.. Enumerable.Range(0, order).Select(i => (long)(2 * (SplitMix64(baseSeed + SplitMixGamma * (ulong)(i + 1)) % halfBound) + 1))];
    }

    /// <summary>Head h of the PLE layer has the (pleIndex·heads + h + 1)-th prime above <paramref name="vocabBase"/> − 1 rows.</summary>
    private static long[] HeadSizes(int vocabBase, int heads, int pleIndex)
    {
        var sizes = new long[heads];
        long prime = vocabBase - 1;
        int skip = pleIndex * heads;
        for (int n = 0; n < skip + heads; n++)
        {
            prime = NextPrime(prime);
            if (n >= skip)
            {
                sizes[n - skip] = prime;
            }
        }

        return sizes;

        static long NextPrime(long p)
        {
            do
            {
                p++;
            }
            while (!IsPrime(p));
            return p;
        }

        static bool IsPrime(long v)
        {
            if (v < 2 || (v % 2 == 0 && v != 2))
            {
                return v == 2;
            }

            for (long d = 3; d * d <= v; d += 2)
            {
                if (v % d == 0)
                {
                    return false;
                }
            }

            return true;
        }
    }

    /// <summary>The final stream mixer (<c>hyper_connection_mixer.*</c> of the text model) → <c>output_hc.*</c>.</summary>
    private static WeightInfo? MapOutputMixer(string name, WeightInfo w)
    {
        const string prefix = "model.hyper_connection_mixer.";
        if (!name.StartsWith(prefix, StringComparison.Ordinal))
        {
            return null;
        }

        return MapHyperConnection("output_hc.", name[prefix.Length..], w);
    }

    private static WeightInfo? MapHyperConnection(string canonicalPrefix, string suffix, WeightInfo w)
    {
        if (suffix == "hc_norm.weight")
        {
            return HfTransforms.NormPlusOne(w, canonicalPrefix + "norm.weight");
        }

        return s_hyperConnectionPlain.TryGetValue(suffix, out var plain) ? HfTransforms.Plain(w, canonicalPrefix + plain) : null;
    }

    /// <summary>Maps a qwen4_exp-only layer tensor; <see cref="TensorOutcome.Unmapped"/> when the tensor is not one.</summary>
    private static TensorOutcome MapQwen4ExpLayerTensor(string prefix, string suffix, WeightInfo w, LayerTargets targets)
    {
        foreach (var (hf, canonical) in new[] { ("attn_hyper_connection.", "attention_hc."), ("mlp_hyper_connection.", "ffn_hc.") })
        {
            if (suffix.StartsWith(hf, StringComparison.Ordinal))
            {
                return Add(targets.Collector, MapHyperConnection(prefix + canonical, suffix[hf.Length..], w));
            }
        }

        if (s_qwen4ExpLayer.TryGetValue(suffix, out var rule))
        {
            targets.Collector.Add(rule.NormPlusOne ? HfTransforms.NormPlusOne(w, prefix + rule.Name) : HfTransforms.Plain(w, prefix + rule.Name));
            return TensorOutcome.Mapped;
        }

        if (NGramShardName().Match(suffix) is { Success: true } shard)
        {
            targets.TableShards.Add(prefix + "ngram.table.weight", int.Parse(shard.Groups[1].Value), w);
            return TensorOutcome.Mapped;
        }

        switch (suffix)
        {
            case "ple.conv1d.weight":
                long channels = w.Shape[0].Fixed;
                targets.Collector.Add(HfTransforms.ToF32(w, prefix + "ngram.conv1d.weight", new Shape(channels, w.Shape.Elements() / channels)));
                return TensorOutcome.Mapped;
            case "ple.ple_embedding.layer_multipliers" or "ple.ple_embedding.ngram_heads_offsets" or "ple.ple_embedding.ngram_heads_vocab_sizes":
                return TensorOutcome.Skipped;   // derived from the config (hyper-parameters)
            case "self_attn.indexer.index_qk_proj.weight":
                long queryRows = (long)targets.Config.Int("indexer_n_heads") * targets.Config.Int("indexer_head_dim");
                targets.Collector.Add(w.RowSlice(prefix + "attention.indexer.q.weight", 0, queryRows));
                targets.Collector.Add(w.RowSlice(prefix + "attention.indexer.k.weight", queryRows, w.Shape[0].Fixed - queryRows));
                return TensorOutcome.Mapped;
            default:
                return TensorOutcome.Unmapped;
        }
    }
}
