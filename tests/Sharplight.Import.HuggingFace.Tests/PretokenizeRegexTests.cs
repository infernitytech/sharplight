using System.Text.Json.Nodes;
using Sharplight.Models;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// tokenizer_config.json's <c>pretokenize_regex</c> overrides tokenizer.json's Split regex, as transformers 5 applies it: Qwen3.5
/// derivatives (Ornith-1.0-9B in the top-20 sweep) ship the Qwen2 regex in tokenizer.json and the mark-aware Qwen3.5 regex in the config,
/// and only the latter matches transformers on Devanagari vowel signs. Built from tiny-random-qwen3.5's tokenizer files with the
/// tokenizer.json regex replaced by the Qwen2 one; skipped when they are missing.
/// </summary>
public sealed class PretokenizeRegexTests
{
    private const string Qwen2Regex = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    private static readonly string Source = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "hf", "tiny-random-qwen3.5");

    private static string PreTokenizer(bool keepConfiguredRegex)
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Source, "tokenizer_config.json")), $"{Source} not present");
        var dir = Directory.CreateTempSubdirectory("pretokenize-regex");
        try
        {
            var tokenizer = JsonNode.Parse(File.ReadAllText(Path.Combine(Source, "tokenizer.json")))!;
            tokenizer["pre_tokenizer"]!["pretokenizers"]![0]!["pattern"]!["Regex"] = Qwen2Regex;
            File.WriteAllText(Path.Combine(dir.FullName, "tokenizer.json"), tokenizer.ToJsonString());
            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(Source, "tokenizer_config.json")))!.AsObject();
            if (!keepConfiguredRegex)
            {
                config.Remove("pretokenize_regex");
            }

            File.WriteAllText(Path.Combine(dir.FullName, "tokenizer_config.json"), config.ToJsonString());
            string configJson = File.ReadAllText(Path.Combine(Source, "config.json"));
            using var hfConfig = HfConfig.Parse(configJson, dir.FullName);
            var md = new Dictionary<string, object>(StringComparer.Ordinal);
            HfTokenizerTranslator.AddTo(md, dir.FullName, hfConfig);
            return (string)md[CanonicalKeys.TokenizerPre];
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public void ConfiguredRegexWins()
    {
        Assert.Equal("qwen35", PreTokenizer(keepConfiguredRegex: true));
    }

    [Fact]
    public void TokenizerJsonRegexWithoutConfiguredOne()
    {
        Assert.Equal("qwen2", PreTokenizer(keepConfiguredRegex: false));
    }
}
