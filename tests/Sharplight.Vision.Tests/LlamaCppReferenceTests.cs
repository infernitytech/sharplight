using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Vision-tower outputs against llama.cpp (<c>llama-mtmd-debug -p encode</c>, which feeds raw float images straight into the
/// encoder and prints the final embeddings' sum and corner values with 4 decimals). Reference numbers were captured from
/// llama.cpp for the converted Qwen3.5-0.8B tower (f32 mmproj) and for the Qwen3.8-27B BF16 mmproj.
/// </summary>
public class LlamaCppReferenceTests
{
    static float[] Cb(int n)
    {
        var v = new float[n * n * 3];
        for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) for (int c = 0; c < 3; c++) v[(y * n + x) * 3 + c] = ((x + y) % 2) != 0 ? 0f : 1f;
        return v;
    }

    /// <summary>The "rainbow" pattern of mtmd-debug (hue by angle, saturation by distance from the centre).</summary>
    static float[] Rainbow(int n)
    {
        var img = new float[n * n * 3];
        float cx = n / 2f, cy = n / 2f, maxDist = MathF.Sqrt(cx * cx + cy * cy);
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float dx = x - cx, dy = y - cy;
                float hue = MathF.Atan2(dy, dx) / (2f * 3.14159265f);
                if (hue < 0) hue += 1f;
                float sat = Math.Min(MathF.Sqrt(dx * dx + dy * dy) / maxDist, 1f);
                float h6 = hue * 6f; int i6 = (int)h6; float f = h6 - i6;
                float p = 1f - sat, q = 1f - sat * f, t = 1f - sat * (1f - f);
                (float r, float g, float b) = (i6 % 6) switch { 0 => (1f, t, p), 1 => (q, 1f, p), 2 => (p, 1f, t), 3 => (p, q, 1f), 4 => (t, p, 1f), _ => (1f, p, q) };
                img[(y * n + x) * 3] = r; img[(y * n + x) * 3 + 1] = g; img[(y * n + x) * 3 + 2] = b;
            }
        return img;
    }

    /// <summary>First/last three values of the first and last output rows and the sum, as printed by llama.cpp.</summary>
    sealed record Expected(float Sum, float[] FirstRowHead, float[] FirstRowTail, float[] LastRowHead, float[] LastRowTail);

    float[] Encode(LlmModel model, float[] image, int n)
    {
        using var enc = model.CreateVisionEncoder();
        return enc.Encode(enc.Preprocessor.Patchify(image, n, n));
    }

    void Check(string what, float[] emb, int rows, int dim, Expected e, double tolerance = 2e-3)
    {
        Assert.Equal(rows * dim, emb.Length);
        double sum = 0, abs = 0; foreach (var v in emb) { sum += v; abs += Math.Abs(v); }
        float[] Head(int r) => emb[(r * dim)..(r * dim + 3)];
        float[] Tail(int r) => emb[(r * dim + dim - 3)..((r + 1) * dim)];
        double worst = 0;
        Console.Error.WriteLine($"  corners ours: {string.Join(" ", new[] { Head(0), Tail(0), Head(rows - 1), Tail(rows - 1) }.Select(a => "[" + string.Join(", ", a.Select(v => v.ToString("F4"))) + "]"))}");
        foreach (var (got, exp) in new[] { (Head(0), e.FirstRowHead), (Tail(0), e.FirstRowTail), (Head(rows - 1), e.LastRowHead), (Tail(rows - 1), e.LastRowTail) })
            for (int i = 0; i < 3; i++) worst = Math.Max(worst, Math.Abs(got[i] - exp[i]));
        Console.Error.WriteLine($"{what}: sum {sum:F4} (llama.cpp {e.Sum:F4}; sum of |x| = {abs:F0}), worst corner-value difference {worst:E2} (llama.cpp prints 4 decimals)");
        // llama.cpp's CPU backend runs attention in f16 and the merger MLP with the tanh GELU: expect ~1e-3 level differences.
        Assert.True(worst < tolerance, $"{what}: corner values differ by {worst}");
        Assert.True(Math.Abs(sum - e.Sum) < 2e-4 * abs, $"{what}: sum {sum} vs {e.Sum}");
    }

    static readonly Expected Cb08 = new(-309.229736f, [-1.7434f, 0.0047f, 0.0853f], [-0.0603f, -0.0621f, 0.1089f], [-1.9890f, 0.0645f, 0.2862f], [-0.2158f, -0.0679f, 0.4822f]);
    static readonly Expected Rainbow08 = new(-222.123230f, [-1.9358f, -0.0911f, 0.1857f], [-0.1916f, -0.0758f, 0.1062f], [-1.6555f, -0.1164f, 0.3913f], [-0.0476f, -0.1362f, 0.4229f]);

    [Fact]
    public void Qwen3_5_0_8b_tower_matches_llama_cpp()
    {
        Assert.SkipUnless(File.Exists(Env.Gguf08) && File.Exists(Env.Mmproj08), "Qwen3.5-0.8B GGUF or converted mmproj not available");
        using var model = LlmModel.Load(Env.Gguf08, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 64, Auxiliary = [(AuxiliaryRoles.Mmproj, Env.Mmproj08)], ImageTokens = new ImageTokenLimits(null, 64) });
        Check("0.8B cb 224", Encode(model, Cb(224), 224), 49, 1024, Cb08);
        Check("0.8B rainbow 224", Encode(model, Rainbow(224), 224), 49, 1024, Rainbow08);
    }

    [Fact]
    public void Qwen3_5_0_8b_hf_tower_matches_llama_cpp()
    {
        Assert.SkipUnless(Directory.Exists(Env.Hf08), "Qwen3.5-0.8B (HF) not available");
        using var model = LlmModel.Load(Env.Hf08, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 64, ImageTokens = new ImageTokenLimits(null, 64) });
        Check("0.8B (HF) rainbow 224", Encode(model, Rainbow(224), 224), 49, 1024, Rainbow08);
    }

    static readonly Expected Rainbow27 = new(-2177.532715f, [0.9753f, -0.0994f, -0.2534f], [1.0165f, -0.9176f, 1.0171f], [0.4478f, 0.1332f, 0.1313f], [0.3458f, -0.1460f, 0.2613f]);

    /// <summary>The 27B tower (27 layers, head size 72, 5120-wide output) from its BF16 mmproj.</summary>
    [Fact]
    public void Qwen3_8_27b_tower_matches_llama_cpp()
    {
        Assert.SkipUnless(File.Exists(Env.Gguf27) && File.Exists(Env.Mmproj27), "Qwen3.8-27B GGUF or mmproj not available");
        using var model = LlmModel.Load(Env.Gguf27, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 64, Auxiliary = [(AuxiliaryRoles.Mmproj, Env.Mmproj27)], ImageTokens = new ImageTokenLimits(null, 64) });
        // llama.cpp's CPU backend multiplies bf16 weights with bf16-rounded activations (and f16 flash attention): it differs from
        // itself by ~2e-3 between flash-attention settings, so allow 1e-2 on these O(1) values.
        Check("27B rainbow 224", Encode(model, Rainbow(224), 224), 49, 5120, Rainbow27, tolerance: 1e-2);
    }
}
