using System.Text.Json;
using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>
/// Survey models that need symbolic/data-dependent shapes, control flow or the newer contrib attention ops (docs/dynamic-shapes.md),
/// compared against onnxruntime. References come from <c>bench/onnx/make_survey_refs.py</c> (models in ~/models/onnx/&lt;name&gt;/, refs in
/// ~/models/onnx/ref/survey); a case is skipped when its manifest or model file is absent.
/// </summary>
public class SurveyModelTests(ITestOutputHelper output)
{
    /// <summary>Per-model settings on top of the manifest.</summary>
    /// <param name="Name">Manifest name in ref/survey.</param>
    /// <param name="Providers">Providers to run on ("cpu", "vulkan").</param>
    /// <param name="RmsTol">Relative RMS criterion instead of the element-wise tolerance (noisy fp16 references, near-tied detections).</param>
    /// <param name="FixBatch">Import with the concrete batch size.</param>
    /// <param name="Pins">Inputs pinned to their reference values at import (they decide output shapes).</param>
    private sealed record Spec(string Name, string[] Providers, double? RmsTol = null, bool FixBatch = false, string[]? Pins = null);

    private static readonly Spec[] s_specs =
    [
        // NonMaxSuppression + TopK over a data-dependent count; two boxes with scores 5e-8 apart swap on CPU.
        new("ssd-10", ["cpu", "vulkan"], RmsTol: 5e-2),
        // NonMaxSuppression + constant-folded TF "arange" Loops; concatenates along the batch axis.
        new("yolov3-12", ["cpu", "vulkan"], FixBatch: true),
        // GroupQueryAttention (rotary, local window) and default-domain SimplifiedLayerNormalization.
        new("gemma3-270m", ["cpu", "vulkan"]),
        new("lfm2-350m", ["cpu", "vulkan"], Pins: ["num_logits_to_keep"]),
        // RotaryEmbedding + MultiHeadAttention with past; fp16 reference noise. The CPU backend runs out of host memory on 3.8B weights.
        new("phi35-mini-q4f16", ["vulkan"], RmsTol: 5e-2),
        // int64 graph input returned as an output.
        new("bertsquad-12", ["cpu", "vulkan"], FixBatch: true),
        // Tile by the batch size, reshapes to expression dims.
        new("deberta-v3-small", ["cpu", "vulkan"]),
    ];

    public static IEnumerable<TheoryDataRow<string, string>> Cases()
        => s_specs.SelectMany(s => s.Providers.Select(p => new TheoryDataRow<string, string>(s.Name, p)));

    [Theory, MemberData(nameof(Cases))]
    public void Survey(string name, string provider)
    {
        var spec = s_specs.Single(s => s.Name == name);
        var c = Load(spec);
        if (c is null)
        {
            Assert.Skip($"survey reference or model missing for '{name}' (bench/onnx/make_survey_refs.py)");
            return;
        }

        var dev = Providers.Get(provider);
        foreach (int batch in c.Batches)
        {
            var inputs = c.RuntimeInputs.Select(k => c.ReadInput(batch, k)).ToList();
            var res = OnnxExec.Run(c.ModelPath, OnnxExec.OptionsFor(c, batch), dev, batch, inputs);
            Assert.Equal(c.Outputs.Length, res.Outputs.Length);
            for (int k = 0; k < c.Outputs.Length; k++)
            {
                var expected = c.ReadOutput(batch, k);
                var cmp = c.RmsTol is { } rt ? Compare.CheckRms(res.Outputs[k], expected, rt) : Compare.Check(res.Outputs[k], expected, c.Rtol, c.Atol);
                output.WriteLine($"{provider} {name} b={batch} out{k} ({c.Outputs[k]}): maxAbs={cmp.MaxAbs:E2} rms={cmp.RmsRel:E2} scale={cmp.Scale:F2} compile={res.CompileMs:F0}ms");
                Assert.True(cmp.Ok, $"{provider} {name} batch {batch} output '{c.Outputs[k]}': maxAbs={cmp.MaxAbs:E3} maxRel={cmp.MaxRel:E3} rms={cmp.RmsRel:E3} at {cmp.Worst}");
            }
        }
    }

    private static RefCase? Load(Spec spec)
    {
        var manifest = Path.Combine(RefCase.RefRoot, "survey", spec.Name + ".json");
        if (!File.Exists(manifest))
        {
            return null;
        }

        using var doc = JsonDocument.Parse(File.ReadAllText(manifest));
        var c = RefCase.TryLoad(manifest, Path.Combine(Path.GetDirectoryName(RefCase.RefRoot)!, doc.RootElement.GetProperty("model").GetString()!));
        if (c is null)
        {
            return null;
        }

        var constants = new Dictionary<string, double[]>(c.Constants);
        foreach (var pin in spec.Pins ?? [])
        {
            int k = Array.FindIndex(c.Inputs, i => i.Name == pin);
            constants[pin] = c.ReadInput(c.Batches[0], k).Select(v => (double)v).ToArray();
        }

        return new RefCase
        {
            Name = c.Name,
            ModelPath = c.ModelPath,
            Prefix = c.Prefix,
            Batches = c.Batches,
            Inputs = c.Inputs,
            Outputs = c.Outputs,
            Rtol = c.Rtol,
            Atol = c.Atol,
            FixBatch = c.FixBatch || spec.FixBatch,
            RmsTol = spec.RmsTol ?? c.RmsTol,
            Mode = c.Mode,
            Constants = constants,
        };
    }
}
