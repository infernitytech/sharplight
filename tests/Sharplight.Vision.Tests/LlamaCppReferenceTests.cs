using Sharplight.Llm;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Vision-tower outputs against llama.cpp (<c>llama-mtmd-debug -p encode</c>, which feeds raw float images straight into the
/// encoder and prints the final embeddings' sum and corner values with 4 decimals). Reference numbers were captured from
/// llama.cpp for the converted Qwen3.5-0.8B tower (f32 mmproj) and for the Qwen3.8-27B BF16 mmproj.
/// </summary>
public class LlamaCppReferenceTests
{
    private static float[] Cb(int n)
    {
        var v = new float[n * n * 3];
        for (int y = 0; y < n; y++)
        {
            for (int x = 0; x < n; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    v[(y * n + x) * 3 + c] = ((x + y) % 2) != 0 ? 0f : 1f;
                }
            }
        }

        return v;
    }

    /// <summary>The "rainbow" pattern of mtmd-debug (hue by angle, saturation by distance from the centre).</summary>
    private static float[] Rainbow(int n)
    {
        var img = new float[n * n * 3];
        float cx = n / 2f;
        float cy = n / 2f;
        float maxDist = MathF.Sqrt(cx * cx + cy * cy);
        for (int y = 0; y < n; y++)
        {
            for (int x = 0; x < n; x++)
            {
                float dx = x - cx;
                float dy = y - cy;
                float hue = MathF.Atan2(dy, dx) / (2f * 3.14159265f);
                if (hue < 0)
                {
                    hue += 1f;
                }

                float sat = Math.Min(MathF.Sqrt(dx * dx + dy * dy) / maxDist, 1f);
                float h6 = hue * 6f;
                int i6 = (int)h6;
                float f = h6 - i6;
                float p = 1f - sat;
                float q = 1f - sat * f;
                float t = 1f - sat * (1f - f);
                (float r, float g, float b) = (i6 % 6) switch { 0 => (1f, t, p), 1 => (q, 1f, p), 2 => (p, 1f, t), 3 => (p, q, 1f), 4 => (t, p, 1f), _ => (1f, p, q) };
                img[(y * n + x) * 3] = r;
                img[(y * n + x) * 3 + 1] = g;
                img[(y * n + x) * 3 + 2] = b;
            }
        }

        return img;
    }

    /// <summary>First/last three values of the first and last output rows and the sum, as printed by llama.cpp.</summary>
    private sealed record Expected(float Sum, float[] FirstRowHead, float[] FirstRowTail, float[] LastRowHead, float[] LastRowTail);

    private float[] Encode(LlmModel model, float[] image, int n)
    {
        using var enc = model.CreateVisionEncoder();
        return enc.Encode(enc.Preprocessor.Patchify(image, n, n));
    }

    private void Check(string what, float[] emb, int rows, int dim, Expected e, double tolerance = 2e-3, double sumTolerance = 2e-4)
    {
        Assert.Equal(rows * dim, emb.Length);
        double sum = 0;
        double abs = 0;
        foreach (var v in emb) { sum += v; abs += Math.Abs(v); }
        float[] Head(int r) => emb[(r * dim)..(r * dim + 3)];
        float[] Tail(int r) => emb[(r * dim + dim - 3)..((r + 1) * dim)];
        double worst = 0;
        Console.Error.WriteLine($"  corners ours: {string.Join(" ", new[] { Head(0), Tail(0), Head(rows - 1), Tail(rows - 1) }.Select(a => "[" + string.Join(", ", a.Select(v => v.ToString("F4"))) + "]"))}");
        foreach (var (got, exp) in new[] { (Head(0), e.FirstRowHead), (Tail(0), e.FirstRowTail), (Head(rows - 1), e.LastRowHead), (Tail(rows - 1), e.LastRowTail) })
        {
            for (int i = 0; i < 3; i++)
            {
                worst = Math.Max(worst, Math.Abs(got[i] - exp[i]));
            }
        }

        Console.Error.WriteLine($"{what}: sum {sum:F4} (llama.cpp {e.Sum:F4}; sum of |x| = {abs:F0}), worst corner-value difference {worst:E2} (llama.cpp prints 4 decimals)");
        // llama.cpp's CPU backend runs attention in f16 and the merger MLP with the tanh GELU: expect ~1e-3 level differences.
        Assert.True(worst < tolerance, $"{what}: corner values differ by {worst}");
        Assert.True(Math.Abs(sum - e.Sum) < sumTolerance * abs, $"{what}: sum {sum} vs {e.Sum}");
    }

    private static readonly Expected Cb08 = new(-309.229736f, [-1.7434f, 0.0047f, 0.0853f], [-0.0603f, -0.0621f, 0.1089f], [-1.9890f, 0.0645f, 0.2862f], [-0.2158f, -0.0679f, 0.4822f]);
    private static readonly Expected Rainbow08 = new(-222.123230f, [-1.9358f, -0.0911f, 0.1857f], [-0.1916f, -0.0758f, 0.1062f], [-1.6555f, -0.1164f, 0.3913f], [-0.0476f, -0.1362f, 0.4229f]);

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

    private static readonly Expected Rainbow27 = new(-2177.532715f, [0.9753f, -0.0994f, -0.2534f], [1.0165f, -0.9176f, 1.0171f], [0.4478f, 0.1332f, 0.1313f], [0.3458f, -0.1460f, 0.2613f]);

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

    // Qwen3-VL-2B (F16 mmproj): rows of 8192 = the 2048-wide embedding followed by the three DeepStack blocks (tower layers 5, 11, 17).
    private static readonly Expected RainbowVl2b = new(2216.004883f, [0.3656f, 0.1490f, -0.7372f], [-0.3639f, -1.0530f, 0.5021f], [0.5162f, 0.3421f, 0.0272f], [0.4566f, -1.2593f, 0.2115f]);

    /// <summary>A DeepStack tower (Qwen3-VL-2B): the merger output and the DeepStack blocks, as llama.cpp concatenates them.</summary>
    [Fact]
    public void Qwen3_vl_2b_deepstack_tower_matches_llama_cpp()
    {
        Assert.SkipUnless(File.Exists(Env.GgufVl2b) && File.Exists(Env.MmprojVl2b), "Qwen3-VL-2B GGUF or mmproj not available");
        using var model = LlmModel.Load(Env.GgufVl2b, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 64, Auxiliary = [(AuxiliaryRoles.Mmproj, Env.MmprojVl2b)], ImageTokens = new ImageTokenLimits(null, 64) });
        Check("Qwen3-VL-2B rainbow 224", Encode(model, Rainbow(224), 224), 49, 8192, RainbowVl2b, tolerance: 1e-2);
    }

    // Qwen2-VL-2B (F16 mmproj from ggml-org, legacy file with swapped MLP names; llama.cpp with -fa off) and Qwen2.5-VL-3B (windowed attention:
    // the 10×10 merged grid is cut into 4×4, 4×2, 2×4 and 2×2 windows), rainbow 280 (100 output rows). These 32-layer towers carry large
    // activations and llama.cpp's CPU backend rounds activations to f16 for the f16 weights: it moves by ~0.01 between its own flash-attention
    // settings and sits ~0.05 from Sharplight (whose Hugging Face towers match transformers to 1e-4, see EndToEndTests), hence the tolerances.
    private static readonly Expected RainbowQwen2Vl2b = new(437.983551f, [-2.5075f, 3.3956f, 2.4169f], [-8.3096f, 4.3038f, -2.5328f], [-2.2033f, -3.1810f, 1.1907f], [2.4686f, -0.0271f, -3.2378f]);
    private static readonly Expected RainbowQwen25Vl3b = new(-1392.166870f, [-2.0526f, -0.1361f, 0.3495f], [-0.5053f, -0.0752f, -0.5379f], [1.3449f, -1.2042f, 2.3733f], [0.9005f, 0.2667f, -1.4410f]);

    /// <summary>Qwen2-VL (LayerNorm, quick GELU, no learned positions) and Qwen2.5-VL (RMS norms, gated MLP, windowed attention) towers.</summary>
    [Theory]
    [InlineData("qwen2-vl")]
    [InlineData("qwen2.5-vl")]
    public void Qwen2_vl_towers_match_llama_cpp(string which)
    {
        bool v25 = which == "qwen2.5-vl";
        string gguf = v25 ? Env.GgufQwen25Vl3b : Env.GgufQwen2Vl2b;
        string mmproj = v25 ? Env.MmprojQwen25Vl3b : Env.MmprojQwen2Vl2b;
        Assert.SkipUnless(File.Exists(gguf) && File.Exists(mmproj), $"{which} GGUF or mmproj not available");
        using var model = LlmModel.Load(gguf, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 64, Auxiliary = [(AuxiliaryRoles.Mmproj, mmproj)], ImageTokens = new ImageTokenLimits(null, 100) });
        Check($"{which} rainbow 280", Encode(model, Rainbow(280), 280), 100, v25 ? 2048 : 1536, v25 ? RainbowQwen25Vl3b : RainbowQwen2Vl2b, tolerance: 0.08, sumTolerance: 1e-3);
    }

    // The tiny random Qwen2-VL / Qwen2.5-VL checkpoints (tools/mk_tiny_qwen2_vl.py) with their towers converted to f32 mmproj files by
    // llama.cpp's converter: rainbow 280, 100 rows of 64.
    private static readonly Expected RainbowTinyQwen2Vl = new(-53.060238f, [-0.0119f, 0.0175f, -0.2446f], [-0.4548f, 0.1236f, 0.1925f], [-0.1137f, 0.0334f, -0.1251f], [-0.3817f, 0.2224f, -0.1487f]);
    private static readonly Expected RainbowTinyQwen25Vl = new(34.210281f, [-0.3658f, -0.1888f, 0.0046f], [0.0849f, 0.0598f, -0.3436f], [-0.1984f, -0.1687f, -0.0830f], [0.1798f, 0.1840f, -0.2941f]);

    /// <summary>The tiny towers in f32 (no rounding on either side): llama.cpp and Sharplight agree to its printed precision.</summary>
    [Theory]
    [InlineData("qwen2-vl-wide")]
    [InlineData("qwen2-5-vl-wide")]
    public void Tiny_qwen2_vl_towers_match_llama_cpp(string which)
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string gguf = Path.Combine(home, $"models/gguf/tiny-random-{which}-f32.gguf");
        string mmproj = Path.Combine(home, $"models/gguf/mmproj-tiny-random-{which}-f32.gguf");
        Assert.SkipUnless(File.Exists(gguf) && File.Exists(mmproj), $"tiny {which} GGUF or mmproj not available");
        using var model = LlmModel.Load(gguf, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 64, Auxiliary = [(AuxiliaryRoles.Mmproj, mmproj)], ImageTokens = new ImageTokenLimits(null, 100) });
        Check($"tiny {which} rainbow 280", Encode(model, Rainbow(280), 280), 100, 64, which == "qwen2-vl-wide" ? RainbowTinyQwen2Vl : RainbowTinyQwen25Vl, tolerance: 2e-4);
    }

    // The tiny random GLM-4.1V (tools/mk_tiny_glm4v.py), its tower converted by llama.cpp b11177 (tools/convert_tiny_mmproj.py, projector type
    // glm4v): rainbow 56 (the 4×4 learned grid itself) and 280 (bicubic resampling), 4 and 100 rows of 64.
    private static readonly Expected RainbowTinyGlm4v56 = new(3.046404f, [0.3155f, 1.2263f, -1.4399f], [0.8930f, 0.5033f, -0.8091f], [-0.0236f, -0.0553f, 0.2907f], [0.0241f, 0.3667f, 0.1091f]);
    private static readonly Expected RainbowTinyGlm4v = new(76.935051f, [0.5760f, 0.3973f, -1.3759f], [-0.2183f, -1.2097f, 1.6504f], [0.1365f, 0.5883f, -0.5596f], [0.4676f, -0.3525f, 0.3442f]);

    /// <summary>GLM-4.1V's tower (patch norm, bicubic position grid, convolution merger, GLM projector) from an f32 mmproj.</summary>
    [Fact]
    public void Tiny_glm4v_tower_matches_llama_cpp()
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string gguf = Path.Combine(home, "models/gguf/tiny-random-glm4v-f32.gguf");
        string mmproj = Path.Combine(home, "models/gguf/mmproj-tiny-random-glm4v-f32.gguf");
        Assert.SkipUnless(File.Exists(gguf) && File.Exists(mmproj), "tiny GLM-4.1V GGUF or mmproj not available");
        using var model = LlmModel.Load(gguf, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 64, Auxiliary = [(AuxiliaryRoles.Mmproj, mmproj)], ImageTokens = new ImageTokenLimits(null, 100) });
        // transformers on the same images gives sums 3.0902 and 76.8782 and the corner values Sharplight prints to 1e-4; llama.cpp (f16
        // attention on the CPU backend) is ~1e-3 away from both.
        Check("tiny glm4v rainbow 56", Encode(model, Rainbow(56), 56), 4, 64, RainbowTinyGlm4v56, tolerance: 3e-3, sumTolerance: 1e-3);
        Check("tiny glm4v rainbow 280", Encode(model, Rainbow(280), 280), 100, 64, RainbowTinyGlm4v, tolerance: 5e-3, sumTolerance: 1e-3);
    }

    // The tiny random PaddleOCR-VL (tools/mk_tiny_paddleocr_vl.py), its tower converted by llama.cpp b11177 (tools/convert_tiny_mmproj.py, projector
    // type paddleocr): rainbow 280, 100 rows of 64.
    private static readonly Expected RainbowTinyPaddleOcr56 = new(-35.144493f, [-0.8637f, 0.4443f, 1.7223f], [2.5447f, -2.8049f, 1.7900f], [-0.9414f, 1.3340f, 2.3642f], [2.5715f, -2.1133f, 1.6070f]);
    private static readonly Expected RainbowTinyPaddleOcr = new(-761.359497f, [-0.7492f, -0.0714f, 1.4031f], [1.6669f, -1.7787f, 2.1794f], [-0.4176f, 1.0500f, 2.0769f], [2.9215f, -2.5334f, 1.1886f]);

    /// <summary>PaddleOCR-VL's SigLIP tower (post-norm, projector merger, 2-D rotary embedding) from an f32 mmproj.</summary>
    [Fact]
    public void Tiny_paddleocr_vl_tower_matches_llama_cpp()
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string gguf = Path.Combine(home, "models/gguf/tiny-random-paddleocr-vl-f32.gguf");
        string mmproj = Path.Combine(home, "models/gguf/mmproj-tiny-random-paddleocr-vl-f32.gguf");
        Assert.SkipUnless(File.Exists(gguf) && File.Exists(mmproj), "tiny PaddleOCR-VL GGUF or mmproj not available");
        using var model = LlmModel.Load(gguf, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 64, Auxiliary = [(AuxiliaryRoles.Mmproj, mmproj)], ImageTokens = new ImageTokenLimits(null, 100) });
        // Sharplight follows transformers (TinyQwenVlImageTests), where llama.cpp deviates twice: its projector uses the tanh GELU (the reference the
        // exact one; ~1e-3 here), and it resamples the learned position grid without align_corners (the reference with), which only matters when
        // the patch grid differs from the learned one: at 56 pixels (the 4×4 grid itself) the towers agree to the GELU difference, at 280 to ~2e-2.
        Check("tiny paddleocr-vl rainbow 56", Encode(model, Rainbow(56), 56), 4, 64, RainbowTinyPaddleOcr56, tolerance: 3e-3, sumTolerance: 1e-3);
        Check("tiny paddleocr-vl rainbow 280", Encode(model, Rainbow(280), 280), 100, 64, RainbowTinyPaddleOcr, tolerance: 3e-2, sumTolerance: 1e-3);
    }
}
