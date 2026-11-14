using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Families of bring-up round 9 (docs/model-support.md) whose layers connect across depth: hierarchical layer cycles (HRM-Text), per-part q/k norms (SenseNova-U1), dense cross-layer connections (Limite), multi-pass streams with a depth memory (Metis 1.6) Tiny random
/// Hugging Face checkpoints come from <c>tools/mk_tiny_decoders.py</c>, their references are greedy ids and prompt logits from transformers
/// (<c>tools/hf_greedy_reference.py --logits</c>); GGUF files are converted from them with llama.cpp's converter (<c>tools/convert_tiny_gguf.py</c>)
/// and checked against llama.cpp 8ca5a80. Every case skips when its files are missing.
/// </summary>
internal static class CrossLayerFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly int[] HrmTextIds = [286, 749, 614, 927, 904, 919, 119, 57, 606, 981, 331, 254, 583, 883, 787, 953, 59, 125, 98, 828, 425, 136, 184, 1006];
    private static readonly int[] LimiteIds = [702, 473, 581, 911, 512, 448, 341, 324, 758, 413, 236, 557, 181, 350, 223, 1019, 273, 351, 860, 383, 664, 444, 676, 634];
    private static readonly int[] Metis16Ids = [71, 933, 52, 845, 206, 731, 158, 380, 52, 700, 206, 942, 645, 845, 577, 338, 509, 963, 845, 432, 697, 926, 249, 11];
    private static readonly int[] SenseNovaU1Ids = [872, 131, 962, 499, 150, 427, 630, 675, 148, 620, 654, 608, 36, 624, 148, 150, 654, 892, 900, 678, 320, 677, 146, 632];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        // HRM-Text (tools/mk_tiny_decoders.py hrm-text): a low and a high stack of 2 layers in 2 high cycles of 2 low passes and a high pass (12 KV
        // slots), weight-less norms, per-channel sigmoid attention gates, the learned low-state init. llama.cpp 8ca5a80 gives the same ids (CPU, Vulkan).
        new("tiny-random-hrm-text HF (hrm_text, transformers)", "hf/tiny-random-hrm-text", Prompt, "ids", HrmTextIds, ""),
        new("tiny-random-hrm-text f32 GGUF (hrm_text, llama.cpp)", "gguf/tiny-random-hrm-text-f32.gguf", Prompt, "ids", HrmTextIds, ""),
        // SenseNova-U1 (tools/mk_tiny_sensenova_u1.py, NEOChatModel layout with the image-generation branch, no transformers class): q/k norms over the
        // temporal and spatial halves of each head, the temporal half rotated. The ids and prompt logits are those of the script's float64 numpy
        // reference of llama.cpp's graph; llama.cpp 8ca5a80 gives the same ids for the converted GGUF (CPU, Vulkan).
        new("tiny-random-sensenova-u1 HF (neo_chat, numpy reference)", "hf/tiny-random-sensenova-u1", Prompt, "ids", SenseNovaU1Ids, ""),
        new("tiny-random-sensenova-u1 f32 GGUF (sensenova_u1, llama.cpp)", "gguf/tiny-random-sensenova-u1-f32.gguf", Prompt, "ids", SenseNovaU1Ids, ""),
        // Limite (tools/mk_tiny_gguf.py limite, remote-code model): value embeddings, exclusive self attention, per-head gates, residual scales,
        // dense connections of three taps into two layers, sliding rope layers around a global NoPE layer, a sigmoid logit cap. The ids are those of
        // tools/ref_limite.py (float64 numpy of llama.cpp's graph); llama.cpp 8ca5a80 gives the same ids (CPU, Vulkan).
        new("tiny-random-limite f32 GGUF (limite, llama.cpp)", "gguf/tiny-random-limite-f32.gguf", Prompt, "ids", LimiteIds, ""),
        // Metis 1.6 (tools/mk_tiny_gguf.py metis16, remote-code model): Mamba-2 / attention / Mamba-2 in two passes over three streams, per-pass
        // stream mixing and attention adapters, the depth memory, routing features into a latent expert router, the continuation head. The ids are
        // those of tools/ref_metis16.py (float64 numpy of llama.cpp's graph); llama.cpp 8ca5a80 gives the same ids (CPU with f32 attention, Vulkan).
        new("tiny-random-metis16 f32 GGUF (metis16, llama.cpp)", "gguf/tiny-random-metis16-f32.gguf", Prompt, "ids", Metis16Ids, ""),
    ];
}
