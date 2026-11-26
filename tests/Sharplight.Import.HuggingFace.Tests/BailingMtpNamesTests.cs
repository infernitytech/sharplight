using System.Runtime.InteropServices;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// Ling 3 / LingSante (inclusionAI/LingSante-3.1-flash, <c>bailing_hybrid</c>) appends its multi-token-prediction layer after the decoder
/// (<c>model.layers.{num_hidden_layers}.</c>) and names the shared head's norm <c>final_layernorm</c> (<c>BailingMoeV3MTPLayer</c>) where
/// DeepSeek writes <c>shared_head.norm</c>. With such a layer's NextN tensors added, the tiny checkpoint imports its trunk unchanged and the
/// NextN tensors below <c>mtp.layers.0.</c>, the norm as the draft head's <c>output_norm</c>.
/// </summary>
public sealed class BailingMtpNamesTests : IDisposable
{
    private const int Hidden = 64;
    private const int TrunkLayers = 6;

    private static readonly string s_fixture = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "hf", "tiny-random-bailing-hybrid");
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "sharplight-bailing-mtp-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir))
        {
            Directory.Delete(_dir, recursive: true);
        }
    }

    [Fact]
    public void ReleasedMtpNames_MapBelowTheDraftModule()
    {
        Assert.SkipUnless(Directory.Exists(s_fixture), $"{s_fixture} not present");
        string mtp = $"model.layers.{TrunkLayers}.";
        var extra = new SafetensorsWriter()
            .AddF32(mtp + "enorm.weight", [Hidden], new float[Hidden])
            .AddF32(mtp + "hnorm.weight", [Hidden], new float[Hidden])
            .AddF32(mtp + "eh_proj.weight", [Hidden, 2 * Hidden], new float[2 * Hidden * Hidden])
            .AddF32(mtp + "final_layernorm.weight", [Hidden], Enumerable.Repeat(0.5f, Hidden).ToArray());
        SafetensorsRenamer.Copy(s_fixture, _dir, name => name, extra);

        var expected = HuggingFaceImporter.Open(s_fixture);
        var actual = HuggingFaceImporter.Open(_dir);
        Assert.All(expected.Weights.Keys, name => Assert.Contains(name, actual.Weights.Keys));
        Assert.Equal(["mtp.layers.0.embedding_norm.weight", "mtp.layers.0.fc.weight", "mtp.layers.0.hidden_norm.weight", "mtp.layers.0.output_norm.weight"],
            actual.Weights.Keys.Except(expected.Weights.Keys).Order(StringComparer.Ordinal));
        float[] norm = MemoryMarshal.Cast<byte, float>(actual.Weights["mtp.layers.0.output_norm.weight"].ReadAll()).ToArray();
        Assert.All(norm, value => Assert.Equal(0.5f, value));
    }
}
