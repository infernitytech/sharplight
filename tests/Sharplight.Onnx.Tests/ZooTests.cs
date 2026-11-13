using System.Globalization;
using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>
/// Architecturally diverse real ONNX models (detection, super-resolution, SAM, diffusion VAE/UNet, whisper decoder, CLIP/SigLIP,
/// ConvNeXt, Swin, wav2vec2, GPT-2, GANs, RAFT, OCR, quantized ...) compared against onnxruntime.
/// References come from <c>scripts/make_zoo_refs.py</c> (models in ~/models/onnx/&lt;name&gt;/, refs in ~/models/onnx/ref/zoo); a case
/// is skipped when its manifest or model file is absent. Environment: SHARPLIGHT_TEST_VK_DEVICE selects the Vulkan device,
/// SHARPLIGHT_ZOO_REPORT=file appends one TSV line per run (name, provider, maxAbs, scale, compile ms, run ms).
/// </summary>
public class ZooTests(ITestOutputHelper output)
{
    public static IEnumerable<TheoryDataRow<string>> Names()
    {
        var names = RefCase.ZooNames().ToList();
        if (names.Count == 0)
        {
            yield return new TheoryDataRow<string>("(no zoo reference data)");
        }

        foreach (var n in names)
        {
            yield return new TheoryDataRow<string>(n);
        }
    }

    private void RunZoo(string name, string provider)
    {
        var c = RefCase.Zoo(name);
        if (c is null) { Assert.Skip($"zoo reference or model missing for '{name}' (scripts/make_zoo_refs.py)"); return; }
        var only = Environment.GetEnvironmentVariable("SHARPLIGHT_ZOO");
        if (!string.IsNullOrEmpty(only) && !only.Split(',').Contains(name)) { Assert.Skip("not selected by SHARPLIGHT_ZOO"); return; }
        var dev = Providers.Get(provider);
        foreach (int batch in c.Batches)
        {
            var inputs = c.RuntimeInputs.Select(k => c.ReadInput(batch, k)).ToList();
            var res = OnnxExec.Run(c.ModelPath, OnnxExec.OptionsFor(c, batch), dev, batch, inputs, timedRuns: 2);
            Assert.Equal(c.Outputs.Length, res.Outputs.Length);
            for (int k = 0; k < c.Outputs.Length; k++)
            {
                if (Environment.GetEnvironmentVariable("SHARPLIGHT_ZOO_DUMP") is { Length: > 0 } dump)
                {
                    var bytes = new byte[res.Outputs[k].Length * 4]; Buffer.BlockCopy(res.Outputs[k], 0, bytes, 0, bytes.Length);
                    File.WriteAllBytes(Path.Combine(dump, $"{name}.{provider}.b{batch}.out{k}.bin"), bytes);
                }
                var actual = res.Outputs[k]; var expected = c.ReadOutput(batch, k);
                if (c.Mode == "scores6" && expected.Length % 6 == 0)
                {
                    actual = Enumerable.Range(0, actual.Length / 6).Select(r => actual[r * 6 + 4]).ToArray();
                    expected = Enumerable.Range(0, expected.Length / 6).Select(r => expected[r * 6 + 4]).ToArray();
                }
                var cmp = c.RmsTol is { } rt ? Compare.CheckRms(actual, expected, rt) : Compare.Check(actual, expected, c.Rtol, c.Atol);
                var line = $"{provider} {name} b={batch} out{k} ({c.Outputs[k]}): maxAbs={cmp.MaxAbs:E2} maxRel={cmp.MaxRel:E2} rms={cmp.RmsRel:E2} scale={cmp.Scale:F2} compile={res.CompileMs:F0}ms run={res.RunMs:F1}ms";
                output.WriteLine(line);
                var report = Environment.GetEnvironmentVariable("SHARPLIGHT_ZOO_REPORT");
                if (!string.IsNullOrEmpty(report))
                {
                    File.AppendAllText(report, string.Join('\t', provider, name, batch.ToString(CultureInfo.InvariantCulture), k.ToString(CultureInfo.InvariantCulture), c.Outputs[k],
                        cmp.MaxAbs.ToString("E3", CultureInfo.InvariantCulture), cmp.Scale.ToString("F2", CultureInfo.InvariantCulture), cmp.RmsRel.ToString("E2", CultureInfo.InvariantCulture), cmp.Ok ? "ok" : "FAIL",
                        res.CompileMs.ToString("F0", CultureInfo.InvariantCulture), res.RunMs.ToString("F1", CultureInfo.InvariantCulture)) + "\n");
                }

                Assert.True(cmp.Ok, $"{provider} {name} batch {batch} output '{c.Outputs[k]}': maxAbs={cmp.MaxAbs:E3} maxRel={cmp.MaxRel:E3} rms={cmp.RmsRel:E3} scale={cmp.Scale:F2} at {cmp.Worst}");
            }
        }
    }

    [Theory, MemberData(nameof(Names))] public void Zoo_Cpu(string name) => RunZoo(name, "cpu");
    [Theory, MemberData(nameof(Names))] public void Zoo_Vulkan(string name) => RunZoo(name, "vulkan");
}
