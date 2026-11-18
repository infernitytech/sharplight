using System.Text.Json;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// VibeVoice ships in two layouts: Microsoft's original checkpoints and transformers' conversions (<c>vibevoice/VibeVoice-1.5B-hf</c>, <c>microsoft/VibeVoice-ASR-HF</c>, which carry their tokenizer
/// files). Both import to the same canonical model: every tensor name of either layout maps to the canonical name set of the other, and the configurations give the same hyper-parameters. Reads
/// only <c>config.json</c> and the safetensors index of each folder under <c>~/models/vibevoice</c> (no weights).
/// </summary>
public sealed class VibeVoiceLayoutTests
{
    private static readonly string s_root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "vibevoice");

    private static bool IsLanguageModel(string name) => VibeVoiceHfShared.LanguageName(name) is not null;

    private static HashSet<string> Language(string folder)
    {
        using var index = JsonDocument.Parse(File.ReadAllText(Path.Combine(s_root, folder, "model.safetensors.index.json")));
        return index.RootElement.GetProperty("weight_map").EnumerateObject().Select(e => VibeVoiceHfShared.LanguageName(e.Name)).OfType<string>().ToHashSet(StringComparer.Ordinal);
    }

    private static HashSet<string> Canonical(string folder, bool recognizer)
    {
        using var index = JsonDocument.Parse(File.ReadAllText(Path.Combine(s_root, folder, "model.safetensors.index.json")));
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var entry in index.RootElement.GetProperty("weight_map").EnumerateObject())
        {
            if (IsLanguageModel(entry.Name) || (recognizer && VibeVoiceHfShared.IsAcousticDecoder(entry.Name)))
            {
                continue;
            }

            string? canonical = VibeVoiceHfShared.CanonicalName(entry.Name);
            Assert.True(canonical is not null, $"{folder}: '{entry.Name}' has no canonical name");
            Assert.True(names.Add(canonical!), $"{folder}: '{entry.Name}' maps to '{canonical}' twice");
        }

        return names;
    }

    private static Dictionary<string, object> Hyper(string folder, IHfArchitectureMapping mapping)
    {
        string dir = Path.Combine(s_root, folder);
        using var config = HfConfig.Parse(File.ReadAllText(Path.Combine(dir, "config.json")), dir);
        Assert.Empty(mapping.Validate(config));
        return mapping.MapHyperParameters(config, new HuggingFaceImportOptions()).ToDictionary(kv => kv.Key, kv => kv.Value);
    }

    private static void AssertSame(Dictionary<string, object> original, Dictionary<string, object> converted, params string[] prefixes)
    {
        foreach (var (key, value) in original)
        {
            // The semantic tokenizer is never sampled: its noise scale is 0 in the original layout and a default in transformers'.
            if (!prefixes.Any(p => key.StartsWith(p, StringComparison.Ordinal)) || key.EndsWith("semantic.vae_std", StringComparison.Ordinal))
            {
                continue;
            }

            Assert.True(converted.TryGetValue(key, out var other), $"'{key}' is missing from the transformers layout");
            string Show(object v) => v is Array a ? string.Join(",", a.Cast<object>()) : Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture)!;
            Assert.True(Show(value) == Show(other!), $"{key}: {Show(value)} (original) vs {Show(other!)} (transformers)");
        }
    }

    /// <summary>The synthesis model: the same canonical tensors and hyper-parameters from both layouts.</summary>
    [Fact]
    public void Synthesis_layouts_agree()
    {
        Assert.SkipUnless(File.Exists(Path.Combine(s_root, "VibeVoice-1.5B", "model.safetensors.index.json")) && File.Exists(Path.Combine(s_root, "VibeVoice-1.5B-hf-meta", "model.safetensors.index.json")), "the VibeVoice folders are not present");
        Assert.Equal(Canonical("VibeVoice-1.5B", false).Order(StringComparer.Ordinal), Canonical("VibeVoice-1.5B-hf-meta", false).Order(StringComparer.Ordinal));
        Assert.Equal(Language("VibeVoice-1.5B").Order(StringComparer.Ordinal), Language("VibeVoice-1.5B-hf-meta").Order(StringComparer.Ordinal));
        var mapping = new VibeVoiceHfMapping();
        var original = Hyper("VibeVoice-1.5B", mapping);
        var converted = Hyper("VibeVoice-1.5B-hf-meta", mapping);
        AssertSame(original, converted, "vibevoice.", "decoder.layer_count", "decoder.hidden_size", "decoder.attention.head_count", "decoder.attention.kv_head_count", "decoder.ffn_hidden_size", "decoder.rope.base");
    }

    /// <summary>The recognizer: the same canonical tensors and hyper-parameters from both layouts.</summary>
    [Fact]
    public void Recognition_layouts_agree()
    {
        Assert.SkipUnless(File.Exists(Path.Combine(s_root, "VibeVoice-ASR", "model.safetensors.index.json")) && File.Exists(Path.Combine(s_root, "VibeVoice-ASR-HF-meta", "model.safetensors.index.json")), "the VibeVoice folders are not present");
        Assert.Equal(Canonical("VibeVoice-ASR", true).Order(StringComparer.Ordinal), Canonical("VibeVoice-ASR-HF-meta", true).Order(StringComparer.Ordinal));
        Assert.Equal(Language("VibeVoice-ASR").Order(StringComparer.Ordinal), Language("VibeVoice-ASR-HF-meta").Order(StringComparer.Ordinal));
        var mapping = new VibeVoiceAsrHfMapping();
        var original = Hyper("VibeVoice-ASR", mapping);
        var converted = Hyper("VibeVoice-ASR-HF-meta", mapping);
        AssertSame(original, converted, "vibevoice_asr.", "decoder.layer_count", "decoder.hidden_size", "decoder.attention.head_count", "decoder.attention.kv_head_count", "decoder.ffn_hidden_size", "decoder.rope.base");
    }
}
