using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Architectures llama.cpp added after the pinned parity commit 8ca5a80 (docs/model-support.md section 26). The tiny checkpoints come from
/// <c>tools/mk_tiny_remote_code.py</c>; their references are greedy ids from transformers running the shipped modeling code
/// (<c>tools/hf_greedy_reference.py</c>), and the GGUF files llama.cpp's converter writes from them give the same text on llama.cpp b12798
/// (9fa7a33) unless a case says otherwise. Every case skips when its files are missing.
/// </summary>
internal static class LlamaCppAdditionsFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    // Reference ids shared by a Hugging Face case and the GGUF converted from it. Declared before All, which reads them.
    private static readonly int[] DomynEdgeIds = [61641, 61641, 61641, 61641, 61641, 61641, 26466, 44786, 17650, 7904, 8735, 47416, 47416, 47416, 23678, 101963,
        101963, 64974, 103280, 15925, 105556, 9700, 9700, 9700];

    // Hunyuan v4 Flash (tools/mk_tiny_hy_v4_flash.py, no transformers class): greedy ids of llama.cpp HEAD a07831a (tools/llamacpp_greedy.c) on the
    // f32 GGUF its converter writes from the checkpoint, up to the tiny tokenizer's end of text (id 0, the 20th); smallest top-2 margin 0.062.
    // 8 indexer heads: with fewer, pools often score exactly 0 (every head's dot negative), and llama.cpp's top-k breaks such ties in no
    // defined order (the smallest pick margin is now 1.4%).
    private static readonly int[] HyV4FlashIds = [793, 48, 335, 991, 340, 772, 162, 899, 51, 988, 131, 773, 523, 262, 177, 831, 513, 756, 899];

    private static readonly int[] Zaya1Ids = [125430, 125430, 112308, 138022, 154165, 97758, 60857, 19136, 111506, 128030, 180860, 106083, 171838, 186001,
        126831, 193772, 33430, 112308, 247982, 243323, 176748, 116330, 94579, 112308];

    private static readonly int[] Raon1Ids = [14698, 125645, 130317, 114135, 209626, 205830, 216895, 106556, 244185, 52926, 120701, 31644, 40155, 124286,
        181149, 60923, 220082, 257274, 244670, 113180, 255343, 91987, 75400, 7044];

    private static readonly int[] Raon1SteerIds = [210247, 144167, 207765, 249624, 225638, 6841, 135515, 88719, 250411, 141806, 206684, 151994, 171587, 92449,
        216070, 143576, 161894, 19857, 52833, 117613, 148731, 125715, 39600, 97346];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // Domyn Edge (domynedge, remote code): sandwich norms, unweighted q norm and weighted k norm, rope only on the sliding-window layers, whose
        // heads are half the size of the full layers' (per-layer head sizes and attention scales). Smallest margin 0.21 (seed 1, weight std 0.08).
        new("tiny-random-domynedge HF (domynedge, remote code)", "hf/tiny-random-domynedge", Prompt, "ids", DomynEdgeIds, ""),
        new("tiny-random-domynedge f32 GGUF (domynedge, llama.cpp)", "gguf/tiny-random-domynedge-hf-f32.gguf", Prompt, "ids", DomynEdgeIds, ""),

        // Hunyuan v4 Flash (hy_v4_flash): sparse layers with the pooled indexer (pools of 4, index_topk 8: the best earlier pool plus the query's
        // own) and sliding-window layers (window 8, twice the heads), sandwich norms, latent MoE, hyper-connections and two Engram modules.
        // F32 KV cache: the pooled keys come from f16 cache rows otherwise, which can swap the indexer's picks.
        // The checkpoint itself (no transformers class: its reference_logits.json holds llama.cpp's prompt logits of the GGUF).
        new("tiny-random-hy-v4-flash HF (hy_v4_flash, llama.cpp reference)", "hf/tiny-random-hy-v4-flash", Prompt, "ids", HyV4FlashIds, "", F32KvCache: true),
        new("tiny-random-hy-v4-flash f32 GGUF (hy_v4_flash, llama.cpp)", "gguf/tiny-random-hy-v4-flash-f32.gguf", Prompt, "ids", HyV4FlashIds, "", F32KvCache: true),

        // ZAYA1 (zaya, transformers): CCA attention (depthwise + grouped q/k convolutions, residual mean-of-heads paths, L2-normalized q/k with a
        // key temperature, delayed values), the MLP router with a router state carried across layers and a skip expert, per-channel residual
        // scales and an input scale (tools/mk_tiny_zaya1.py).
        // Smallest greedy margin 0.087; the router's top-1 has near-ties that f16 KV rounding flips on the CPU, so the caches stay f32.
        // transformers resets the skip expert's class bias to -1 on load (as stored in ZAYA1-8B), so the skip expert never wins here.
        new("tiny-random-zaya1 HF (zaya, transformers)", "hf/tiny-random-zaya1", Prompt, "ids", Zaya1Ids, "", F32KvCache: true),
        new("tiny-random-zaya1 f32 GGUF (zaya1, llama.cpp)", "gguf/tiny-random-zaya1-f32.gguf", Prompt, "ids", Zaya1Ids, "", F32KvCache: true),
        // The same with the skip expert's class bias 0 (TINY_SKIP_BIAS=0): it wins for part of the tokens and zeroes their MoE output. Only
        // llama.cpp honours the stored bias, so the reference is the text of llama.cpp b12798's llama-simple (which, unlike llama-completion,
        // accepts the trailing <|im_end|> the tokenizer adds); it leaves the plain model's text after 12 tokens.
        new("tiny-random-zaya1-skip f32 GGUF (zaya1 skip expert, llama.cpp)", "gguf/tiny-random-zaya1-skip-f32.gguf", Prompt, "text", [],
            " Herzog Herzogक्रमा सलाहकारстинreservedutzer اللہ Recommendation GLC看一下migration unt ব্যবসার fearedotusの関係 ร่วมSheikhSheikh coloridoக்குற прикла<unused5720>", Tokens: 24, F32KvCache: true),
        // Raon-1 mini (raon1_mini, KRAFTON's remote code; tools/mk_tiny_raon1.py): Raon delta attention (write gate, bounded decay, separate q/k/v
        // short convolutions), NoPE gated latent attention on the 1-based layers 3 and 6, a dense SiTU first layer and sigmoid-routed SiTU MoE
        // with a shared expert, four Raon_mHC streams with the Helmert-chart mixing; the real tokenizer (pre-tokenizer "raon", NFC).
        // Smallest greedy margin 0.034.
        new("tiny-random-raon1-mini HF (raon1_mini, remote code)", "hf/tiny-random-raon1-mini", Prompt, "ids", Raon1Ids, "", Tokens: 24),
        new("tiny-random-raon1-mini f32 GGUF (raon1, llama.cpp)", "gguf/tiny-random-raon1-mini-f32.gguf", Prompt, "ids", Raon1Ids, "", Tokens: 24),
        // The same with reasoning-effort steering vectors on layers 2 and 4 at the default effort (TINY_SEED=1 TINY_STEERING=1; HF only, llama.cpp's
        // converter does not take them). Smallest greedy margin 0.136.
        new("tiny-random-raon1-mini-steer HF (raon1_mini, reasoning-effort steering, remote code)", "hf/tiny-random-raon1-mini-steer", Prompt, "ids",
            Raon1SteerIds, "", Tokens: 24),
    ];
}
