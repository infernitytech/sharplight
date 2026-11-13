using Sharplight.Gguf;
using Sharplight.Models;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.HuggingFace.Tests;

/// <summary>
/// Ternary-Bonsai-2 ships the same ternary weights in Q2_0, PQ2_0 and PTQ1_0 packings, stored in a Hadamard-rotated basis
/// (<c>prism.hadamard.*</c>). After import the three must yield identical dense canonical weights (rotation undone), and the
/// tied output head equals the embedding. Skips when the files are absent.
/// </summary>
public class BonsaiHadamardTests
{
    private static readonly string Dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "gguf", "Ternary-Bonsai-2-2B");
    private static readonly string[] Files = ["Ternary-Bonsai-2-2B-Q2_0.gguf", "Ternary-Bonsai-2-2B-PQ2_0.gguf", "Ternary-Bonsai-2-2B-PTQ1_0.gguf"];

    private static byte[] Read(WeightInfo w) => w.ReadAll();

    [Fact]
    public void Packings_Import_To_Identical_Dense_Weights()
    {
        var paths = Files.Select(f => Path.Combine(Dir, f)).ToArray();
        if (!paths.All(File.Exists))
        {
            Assert.Skip("Ternary-Bonsai-2-2B GGUF files not present");
            return;
        }
        var sources = paths.Select(p => GgufImporter.Open(p)).ToArray();
        try
        {
            foreach (var s in sources)
            {
                Assert.Equal("qwen3_5", s.Architecture);
            }

            foreach (var name in new[] { "layers.0.linear_attention.qkv.weight", "layers.0.mlp.down.weight", "layers.3.attention.o.weight" })
            {
                var w0 = sources[0].Weights[name];
                Assert.Equal(DataType.F16, w0.Type);   // rotated back to dense f16
                var b0 = Read(w0);
                for (int i = 1; i < sources.Length; i++)
                {
                    Assert.True(b0.AsSpan().SequenceEqual(Read(sources[i].Weights[name])), $"{name} differs between packings");
                }
                // A rotated-back ternary weight is no longer ternary but must be non-trivial and finite.
                var h = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Half>(b0.AsSpan(0, 4096));
                Assert.Contains(h.ToArray(), v => v != (Half)0);
                Assert.All(h.ToArray(), v => Assert.True(Half.IsFinite(v)));
            }
            Assert.DoesNotContain("lm_head.weight", sources[0].Weights.Keys);   // tied head: shares the (rotated back) embedding
            Assert.Equal(DataType.F16, sources[0].Weights["token_embedding.weight"].Type);
        }
        finally { foreach (var s in sources)
            {
                foreach (var r in s.OwnedResources)
                {
                    r.Dispose();
                }
            }
        }
    }
}
