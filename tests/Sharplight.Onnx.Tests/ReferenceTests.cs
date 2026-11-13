using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>
/// End-to-end numeric comparison against onnxruntime. Reference data comes from
/// <c>.work/onnx_ref/make_synthetic.py</c> and <c>make_model_refs.py</c> (written to ~/models/onnx/ref); cases skip when absent.
/// </summary>
public class ReferenceTests(ITestOutputHelper output)
{
    public static IEnumerable<TheoryDataRow<string>> SyntheticCases()
    {
        var names = RefCase.SyntheticNames().ToList();
        if (names.Count == 0)
        {
            yield return new TheoryDataRow<string>("(no reference data)");
        }

        foreach (var n in names)
        {
            yield return new TheoryDataRow<string>(n);
        }
    }

    private void RunCase(RefCase c, string provider)
    {
        IComputeDevice dev = Providers.Get(provider);
        foreach (int batch in c.Batches)
        {
            var inputs = Enumerable.Range(0, c.Inputs.Length).Select(k => c.ReadInput(batch, k)).ToList();
            OnnxExec.Result res;
            try { res = OnnxExec.Run(c.ModelPath, OnnxExec.OptionsFor(c, batch), dev, batch, inputs); }
            catch (NotSupportedException e) when (provider != "cpu")
            {
                Assert.Skip($"{provider}: {e.Message}");
                return;
            }
            Assert.Equal(c.Outputs.Length, res.Outputs.Length);
            for (int k = 0; k < c.Outputs.Length; k++)
            {
                var cmp = Compare.Check(res.Outputs[k], c.ReadOutput(batch, k), c.Rtol, c.Atol);
                output.WriteLine($"{provider} {c.Name} b={batch} out{k} ({c.Outputs[k]}): maxAbs={cmp.MaxAbs:E2} maxRel={cmp.MaxRel:E2} scale={cmp.Scale:F2} compile={res.CompileMs:F0}ms");
                Assert.True(cmp.Ok, $"{provider} {c.Name} batch {batch} output '{c.Outputs[k]}': maxAbs={cmp.MaxAbs:E3} maxRel={cmp.MaxRel:E3} at {cmp.Worst}");
            }
        }
    }

    [Theory, MemberData(nameof(SyntheticCases))]
    public void Synthetic_Cpu(string name)
    {
        var c = RefCase.Synthetic(name);
        if (c is null) { Assert.Skip("reference data missing; run .work/onnx_ref/make_synthetic.py"); return; }
        RunCase(c, "cpu");
    }

    [Theory, MemberData(nameof(SyntheticCases))]
    public void Synthetic_Vulkan(string name)
    {
        var c = RefCase.Synthetic(name);
        if (c is null) { Assert.Skip("reference data missing; run .work/onnx_ref/make_synthetic.py"); return; }
        RunCase(c, "vulkan");
    }

    // ---- real models ---------------------------------------------------------------------------------
    private void RunModel(string refName, string onnxFile, string provider, bool top1 = true)
    {
        var c = RefCase.Model(refName, onnxFile);
        if (c is null) { Assert.Skip($"{onnxFile}.onnx or its reference data is missing (see .work/onnx_ref/make_model_refs.py)"); return; }
        if (refName == "yolov10n") { YoloScores(c, provider); return; }
        RunCase(c, provider);
        if (!top1)
        {
            return;
        }

        var dev = Providers.Get(provider);
        foreach (int batch in c.Batches)
        {
            var res = OnnxExec.Run(c.ModelPath, OnnxExec.OptionsFor(c, batch), dev, batch, [c.ReadInput(batch, 0)]);
            var exp = c.ReadOutput(batch, 0);
            int per = exp.Length / batch;
            for (int b = 0; b < batch; b++)
            {
                Assert.Equal(Compare.ArgMax(exp.AsSpan(b * per, per)), Compare.ArgMax(res.Outputs[0].AsSpan(b * per, per)));
            }
        }
    }

    /// <summary>
    /// yolov10n on random input yields almost-tied detections, so class ids / boxes of tied rows are order dependent;
    /// compare the (sorted) score column and require most rows to match exactly.
    /// </summary>
    private void YoloScores(RefCase c, string provider)
    {
        var dev = Providers.Get(provider);
        var inputs = new List<float[]> { c.ReadInput(1, 0) };
        OnnxExec.Result res;
        try { res = OnnxExec.Run(c.ModelPath, OnnxExec.OptionsFor(c, 1), dev, 1, inputs); }
        catch (NotSupportedException e) when (provider != "cpu") { Assert.Skip($"{provider}: {e.Message}"); return; }
        var exp = c.ReadOutput(1, 0); var act = res.Outputs[0];
        int rows = exp.Length / 6, same = 0; double maxScore = 0;
        for (int r = 0; r < rows; r++)
        {
            maxScore = Math.Max(maxScore, Math.Abs(exp[r * 6 + 4] - act[r * 6 + 4]));
            bool eq = true; for (int k = 0; k < 6; k++)
            {
                eq &= Math.Abs(exp[r * 6 + k] - act[r * 6 + k]) <= 1e-3 * Math.Max(1, Math.Abs(exp[r * 6 + k]));
            }

            if (eq)
            {
                same++;
            }
        }
        output.WriteLine($"{provider} yolov10n: score maxAbs={maxScore:E2}, identical rows {same}/{rows}");
        Assert.True(maxScore < 1e-4, $"score column differs by {maxScore}");
        Assert.True(same >= rows * 0.8, $"only {same}/{rows} rows identical");
    }

    [Fact] public void ResNet50_Cpu() => RunModel("resnet50-v1-12", "resnet50-v1-12", "cpu");
    [Fact] public void ResNet50_Vulkan() => RunModel("resnet50-v1-12", "resnet50-v1-12", "vulkan");
    [Fact] public void MobileNetV2_Cpu() => RunModel("mobilenetv2-12", "mobilenetv2-12", "cpu");
    [Fact] public void MobileNetV2_Vulkan() => RunModel("mobilenetv2-12", "mobilenetv2-12", "vulkan");
    [Fact] public void DeiTTiny_Cpu() => RunModel("deit-tiny", "deit-tiny", "cpu");
    [Fact] public void DeiTTiny_Vulkan() => RunModel("deit-tiny", "deit-tiny", "vulkan");
    [Fact] public void ViTBase_Cpu() => RunModel("vit-base-patch16-224", "vit-base-patch16-224", "cpu");
    [Fact] public void ViTBase_Vulkan() => RunModel("vit-base-patch16-224", "vit-base-patch16-224", "vulkan");

    // ---- additional real models (best effort; each skips when its file/reference is absent) -----------------------
    public static IEnumerable<TheoryDataRow<string>> ExtraModels() => new[]
        { "whisper-tiny-encoder", "yolov10n", "segformer-b0", "minilm-l6", "squeezenet1.1-7", "mnist-12", "fcn-resnet50-12",
          "distilbert-base-uncased", "t5-small-encoder", "depth-anything-small", "efficientnet-lite4-11" }.Select(n => new TheoryDataRow<string>(n));

    [Theory, MemberData(nameof(ExtraModels))] public void ExtraModel_Cpu(string name) => RunModel(name, name, "cpu", top1: false);
    [Theory, MemberData(nameof(ExtraModels))] public void ExtraModel_Vulkan(string name) => RunModel(name, name, "vulkan", top1: false);
}
