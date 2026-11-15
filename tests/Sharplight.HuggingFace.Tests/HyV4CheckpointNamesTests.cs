using Xunit;

namespace Sharplight.HuggingFace.Tests;

/// <summary>
/// The released Hunyuan v4 checkpoint (tencent/Hy4-preview) spells the hyper-connection tensors <c>hc_attn_layer.hc_pre.hc_fn</c> and
/// <c>model.hc_head.hc_head_fn</c> where the tiny fixture (and earlier conversions) have <c>hc_pre.hc_attn_layer.hc_fn</c> and
/// <c>model.hc_pre.hc_head.hc_head_fn</c>, and it keeps an MTP layer below <c>model.mtp_layers.</c>: renamed that way, and with an MTP
/// tensor added, the tiny checkpoint imports to exactly the same canonical weights.
/// </summary>
public sealed class HyV4CheckpointNamesTests : IDisposable
{
    private static readonly string s_fixture = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "hf", "tiny-random-hy-v4");
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "sharplight-hyv4-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir))
        {
            Directory.Delete(_dir, recursive: true);
        }
    }

    [Fact]
    public void ReleasedNames_ImportLikeTheFixture()
    {
        Assert.SkipUnless(Directory.Exists(s_fixture), $"{s_fixture} not present");
        static string Released(string name) => name
            .Replace("hc_pre.hc_attn_layer.", "hc_attn_layer.hc_pre.", StringComparison.Ordinal)
            .Replace("hc_pre.hc_mlp_layer.", "hc_mlp_layer.hc_pre.", StringComparison.Ordinal)
            .Replace("model.hc_pre.hc_head.", "model.hc_head.", StringComparison.Ordinal);
        var mtp = new SafetensorsWriter().AddF32("model.mtp_layers.0.enorm.weight", [64], new float[64]);
        SafetensorsRenamer.Copy(s_fixture, _dir, Released, mtp);

        var expected = HuggingFaceImporter.Open(s_fixture);
        var actual = HuggingFaceImporter.Open(_dir);
        Assert.Contains(expected.Weights.Keys, k => k.Contains("_hc.", StringComparison.Ordinal));
        Assert.Equal(expected.Weights.Keys.Order(StringComparer.Ordinal), actual.Weights.Keys.Order(StringComparer.Ordinal));
        foreach (var (name, weight) in expected.Weights)
        {
            var other = actual.Weights[name];
            Assert.Equal(weight.Shape, other.Shape);
            Assert.Equal(weight.Type, other.Type);
            Assert.Equal(weight.ReadAll(), other.ReadAll());
        }
    }
}
