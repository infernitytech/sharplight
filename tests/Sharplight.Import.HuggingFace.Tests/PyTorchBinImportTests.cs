using Sharplight.Import.PyTorch;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// A model directory with <c>pytorch_model.bin</c> shards (written by torch.save, tools/mk-pytorch-bin-model.py from
/// tiny-random-OlmoForCausalLM) must import to exactly the weights of its safetensors original. Skips when the directories are absent.
/// </summary>
public sealed class PyTorchBinImportTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/hf");
    private static readonly string s_original = Path.Combine(Models, "tiny-random-OlmoForCausalLM");

    [Theory]
    [InlineData("tiny-random-Olmo-single")]
    [InlineData("tiny-random-Olmo-sharded")]
    public void BinCheckpointImportsLikeSafetensors(string name)
    {
        string dir = Path.Combine(Models, "pytorch-bin", name);
        if (!Directory.Exists(dir) || !Directory.Exists(s_original))
        {
            return;
        }

        Assert.True(new HuggingFaceImporter().CanOpen(dir));
        var expected = HuggingFaceImporter.Open(s_original);
        var actual = HuggingFaceImporter.Open(dir);
        try
        {
            Assert.Equal(expected.Weights.Keys.Order(StringComparer.Ordinal), actual.Weights.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, want) in expected.Weights)
            {
                var got = actual.Weights[key];
                Assert.Equal(want.Type, got.Type);
                Assert.Equal(want.Shape, got.Shape);
                var a = new byte[want.Source.ByteLength];
                var b = new byte[got.Source.ByteLength];
                want.Source.Read(0, a);
                got.Source.Read(0, b);
                Assert.True(a.AsSpan().SequenceEqual(b), key);
            }
        }
        finally
        {
            foreach (var resource in expected.OwnedResources.Concat(actual.OwnedResources))
            {
                resource.Dispose();
            }
        }
    }

    [Fact]
    public void MaliciousBinCheckpointIsRefusedWithoutExecution()
    {
        string dir = Path.Combine(Path.GetTempPath(), "sl-hf-evil-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir, "config.json"), """{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}""");
            byte[] evil = [0x80, 2, .. "cos\nsystem\n"u8, (byte)'X', 4, 0, 0, 0, .. "echo"u8, (byte)'R', (byte)'.'];
            File.WriteAllBytes(Path.Combine(dir, "pytorch_model.bin"), evil);
            Assert.Throws<UnsafePickleException>(() => HuggingFaceImporter.Open(dir));
            Assert.Throws<UnsafePickleException>(() => SafetensorsCheckpoint.Open(dir));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
