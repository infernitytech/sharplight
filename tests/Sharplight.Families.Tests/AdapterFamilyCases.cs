using static Sharplight.Families.Tests.FamilyGenerationTests;

namespace Sharplight.Families.Tests;

/// <summary>
/// Families with per-token adapters (bring-up round 9). Granite Switch has no transformers class: the tiny checkpoint of
/// <c>tools/mk_tiny_granite_switch.py</c> (published layout, " France" activating the second adapter) is converted with llama.cpp's converter
/// (<c>tools/convert_tiny_gguf.py</c>); the ids and the prompt logits (<c>reference_logits.json</c>) are those of <c>tools/ref_granite_switch.py</c>
/// (float64 numpy of llama.cpp's graph), which llama.cpp 8ca5a80 gives on the CPU and on Vulkan, with and without flash attention. Every case
/// skips when its files are missing.
/// </summary>
internal static class AdapterFamilyCases
{
    private const string Prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art";

    private static readonly int[] GraniteSwitchIds =
        [4444, 10733, 12553, 46074, 36154, 39915, 43177, 43177, 31758, 24034, 23655, 35771, 45225, 14887, 15693, 42048, 35048, 7079, 10544, 42535, 38227, 17044, 27956, 5510];

    /// <summary>The cases, appended to the case list of <see cref="FamilyGenerationTests"/>.</summary>
    public static readonly Case[] All =
    [
        new("tiny-random-granite-switch HF (granite_switch, llama.cpp graph reference)", "hf/tiny-random-granite-switch", Prompt, "ids", GraniteSwitchIds, ""),
        new("tiny-random-granite-switch f32 GGUF (graniteswitch, llama.cpp)", "gguf/tiny-random-granite-switch-f32.gguf", Prompt, "ids", GraniteSwitchIds, ""),
    ];
}
