using System.Text.Json;
using Sharplight.Llm;

namespace Sharplight.Vision.Tests;

/// <summary>
/// The Gemma 4 vision tower (mmproj projector type <c>gemma4v</c>) of gemma-4-31B-it against transformers' <c>Gemma4VisionModel</c> and
/// <c>Gemma4MultimodalEmbedder</c> (float32, CPU; reference files written by a script that runs them on the processor's patches). Skipped
/// when the model files or the reference are not available. The directory comes from SHARPLIGHT_GEMMA4_31B, else models/hot-a/gemma-4-31b-gguf.
/// </summary>
public class Gemma4TowerTests
{
    private static readonly string s_dir = Environment.GetEnvironmentVariable("SHARPLIGHT_GEMMA4_31B")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/hot-a/gemma-4-31b-gguf");

    [Fact]
    public void Gemma4_31b_tower_matches_transformers()
    {
        string gguf = Path.Combine(s_dir, "gemma-4-31B-it-Q4_K_M.gguf");
        string mmproj = Path.Combine(s_dir, "mmproj-F16.gguf");
        string prefix = Path.Combine(s_dir, "reference", "tower");
        Assert.SkipUnless(File.Exists(gguf) && File.Exists(mmproj) && File.Exists(prefix + ".json"), "Gemma 4 31B GGUF, mmproj or reference not available");
        var meta = JsonDocument.Parse(File.ReadAllText(prefix + ".json")).RootElement;
        int height = meta.GetProperty("height").GetInt32();
        int width = meta.GetProperty("width").GetInt32();
        int tokens = meta.GetProperty("tokens").GetInt32();
        int dim = meta.GetProperty("dim").GetInt32();

        // The reference image in [0, 1]; the tower's normalization is (x - 0.5) / 0.5, the 2x - 1 of the model code.
        var image = Env.ReadFloats(prefix + ".image.bin");
        for (int i = 0; i < image.Length; i++)
        {
            image[i] = (image[i] - 0.5f) / 0.5f;
        }

        var reference = Env.ReadFloats(prefix + ".emb.bin");
        using var model = LlmModel.Load(TestHost.Loader, gguf, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 64, Auxiliary = { [AuxiliaryRoles.Mmproj] = mmproj } });
        using var encoder = model.CreateVisionEncoder();
        var pre = encoder.Preprocessor.Patchify(image, width, height);
        Assert.Equal(tokens, pre.TokenCount);
        var ours = encoder.Encode(pre);
        Assert.Equal(tokens * dim, ours.Length);

        double worstCosine = 1;
        double worstRelative = 0;
        for (int r = 0; r < tokens; r++)
        {
            double dot = 0;
            double na = 0;
            double nb = 0;
            double diff = 0;
            for (int c = 0; c < dim; c++)
            {
                double a = ours[r * dim + c];
                double b = reference[r * dim + c];
                dot += a * b;
                na += a * a;
                nb += b * b;
                diff += (a - b) * (a - b);
            }

            worstCosine = Math.Min(worstCosine, dot / Math.Sqrt(na * nb));
            worstRelative = Math.Max(worstRelative, Math.Sqrt(diff / nb));
        }

        Console.Error.WriteLine($"gemma4 tower: {tokens} tokens, worst row cosine {worstCosine:F6}, worst relative error {worstRelative:E2}");
        Assert.True(worstCosine > 0.999, $"worst row cosine {worstCosine}");
    }
}
