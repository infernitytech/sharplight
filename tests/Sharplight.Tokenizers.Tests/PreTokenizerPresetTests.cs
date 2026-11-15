using Sharplight.Tokenizers.Bpe;
using Xunit;

namespace Sharplight.Tokenizers.Tests;

/// <summary>
/// Every byte-level BPE <c>tokenizer.ggml.pre</c> value of llama.cpp (8ca5a80, <c>llama-vocab.cpp</c>) resolves to a preset: real GGUFs with
/// a preset Sharplight did not know (DeepSeek-V4's <c>joyai-llm</c>, MiniCPM 5, AFMoE, ...) imported but failed when the tokenizer was built.
/// </summary>
public class PreTokenizerPresetTests
{
    /// <summary>llama.cpp's byte-level BPE presets (the ▁-vocabulary ones, <c>gemma4</c>, <c>sarvam-moe</c>, <c>whitespace</c>, ..., are not byte-level).</summary>
    public static TheoryData<string> LlamaCppPresets =>
    [
        "default", "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "jina-v5-nano", "deepseek-llm", "deepseek-coder",
        "deepseek-v3", "youtu", "falcon", "mpt", "starcoder", "gpt-2", "phi-2", "jina-es", "jina-de", "gigachat", "jina-v2-es", "jina-v2-de", "a.x-4.0",
        "mellum", "modern-bert", "jina-v1-en", "jina-v2-code", "roberta-bpe", "refact", "command-r", "qwen2", "deepseek-r1-qwen", "kormo", "megrez",
        "qwen35", "stablelm2", "olmo", "dbrx", "smaug-bpe", "poro-chat", "chatglm-bpe", "viking", "jais", "tekken", "tekken-num3", "smollm", "codeshell",
        "bloom", "gpt3-finnish", "exaone", "exaone4", "exaone-moe", "chameleon", "minerva-7b", "gpt-4o", "llama4", "kanana2", "superbpe", "trillion",
        "granite-docling", "bailingmoe", "bailingmoe2", "llada-moe", "seed-coder", "hunyuan", "hunyuan-dense", "joyai-llm", "hy_v4", "kimi-k2", "grok-2",
        "afmoe", "minimax-m2", "solar-open", "minicpm5", "tiny_aya", "cohere2moe", "spark2_5", "jais-2", "granite-embed-multi-97m", "ufakzeka", "laguna",
        "mellum2", "k2-horizon", "nemotron-3.5-embed", "talkie", "flint", "glm4", "f2llmv2",
    ];

    [Theory]
    [MemberData(nameof(LlamaCppPresets))]
    public void LlamaCppPresetResolves(string pre)
    {
        var preset = PreTokenizerPreset.Resolve(pre);
        Assert.NotEmpty(Split(preset, "The capital of France is Paris. 1234567 ünïcödé 日本語 \n\n  spaces"));
    }

    [Theory]
    [InlineData("afmoe", "1234567", new[] { "1", "234", "567" })]
    [InlineData("tekken-num3", "1234567", new[] { "1", "234", "567" })]
    [InlineData("tiny_aya", "1234567", new[] { "1", "234", "567" })]
    [InlineData("superbpe", "1234567", new[] { "1", "234", "567" })]
    [InlineData("minicpm5", "12345", new[] { "123", "45" })]
    [InlineData("deepseek-coder", "ab12", new[] { "ab", "1", "2" })]
    [InlineData("ufakzeka", "Ankara'da", new[] { "Ankara", "'da" })]
    public void DigitAndWordGroups(string pre, string text, string[] pieces)
        => Assert.Equal(pieces, Split(PreTokenizerPreset.Resolve(pre), text));

    private static List<string> Split(PreTokenizerPreset preset, string text)
    {
        // The tests use text without surrogate pairs: one code point per char.
        int[] cpts = [.. text.Select(c => (int)c)];
        int[] starts = [.. Enumerable.Range(0, text.Length + 1)];
        var lengths = new List<int>();
        preset.Splitter.Split(new SplitInput(text, cpts, starts), lengths);
        var pieces = new List<string>();
        int at = 0;
        foreach (int length in lengths)
        {
            pieces.Add(text.Substring(at, length));
            at += length;
        }

        Assert.Equal(text.Length, at);
        return pieces;
    }
}
