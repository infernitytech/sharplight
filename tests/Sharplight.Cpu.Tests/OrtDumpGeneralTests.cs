using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>Writes inputs/outputs of selected ops to $SHARPLIGHT_ORT_DUMP for the onnxruntime comparison script (no-op otherwise).</summary>
public class OrtDumpGeneralTests
{
    private static void Save(string dir, string name, string suffix, Array a) =>
        File.WriteAllBytes(Path.Combine(dir, $"{name}.{suffix}.bin"), MemoryMarshal.AsBytes(a is float[] f ? f.AsSpan() : a is int[] i ? i.AsSpan().Cast() : Span<float>.Empty).ToArray());

    [Fact]
    public void Dump()
    {
        var dir = Environment.GetEnvironmentVariable("SHARPLIGHT_ORT_DUMP");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }

        dir = Path.Combine(dir, "general");
        Directory.CreateDirectory(dir);
        void Run(string name, Func<TestGraph, (Value[] ins, Value[] outs)> build, params Array[] inputs)
        {
            var tg = new TestGraph();
            var (ins, outs) = build(tg);
            for (int i = 0; i < outs.Length; i++)
            {
                tg.B.Output("o" + i, outs[i]);
            }

            using var m = Kit.Compile(tg.B.Build());
            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] is float[] f)
                {
                    Kit.Set(m, ins[i], f);
                }
                else
                {
                    Kit.Set(m, ins[i], (int[])inputs[i]);
                }

                Save(dir, name, "in" + i, inputs[i]);
            }
            var p = Kit.P(1);
            m.Run(p);
            for (int i = 0; i < outs.Length; i++)
            {
                if (outs[i].Type == DataType.I32)
                {
                    Save(dir, name, "out" + i, Kit.OutI(m, "o" + i, p));
                }
                else
                {
                    Save(dir, name, "out" + i, Kit.Out(m, "o" + i, p));
                }
            }
        }
        foreach (var mode in new[] { "linear", "nearest", "cubic" })
        {
            foreach (var pad in new[] { "zeros", "border", "reflection" })
            {
                foreach (var ac in new[] { false, true })
        {
            var x = Kit.Rand(2 * 5 * 6, 201);
            var g = Kit.Rand(4 * 5 * 2, 202, 1.4f);
            if (mode == "nearest")
                    {
                        for (int i = 0; i < 8; i++)
                        {
                            g[i] = (i % 2 == 0 ? -1f : 1f) * (i / 2 * 0.25f); // exact ties
                        }
                    }

                    Run($"grid_{mode}_{pad}_{(ac ? 1 : 0)}", tg => { var xi = tg.InF("x", 1, 2, 5, 6); var gi = tg.InF("g", 1, 4, 5, 2); return ([xi, gi], [tg.B.Emit1("GridSample", [xi, gi], GeneralKit.A(("mode", mode), ("padding", pad), ("align_corners", ac)))]); }, x, g);
        }
            }
        }

        foreach (var largest in new[] { true, false })
        {
            var x = Kit.Rand(3 * 10, 203).Select(v => (float)Math.Round(v * 3)).ToArray();
            Run($"topk_{(largest ? 1 : 0)}", tg => { var xi = tg.InF("x", 3, 10); var o = tg.B.Emit("TopK", [xi], GeneralKit.A(("k", 4L), ("largest", largest))); return ([xi], [o[0], o[1]]); }, x);
        }
        { var a = Kit.Rand(40, 204, 7f); var b = Kit.Rand(40, 205, 3f); Run("mod_fmod", tg => { var ai = tg.InF("a", 40); var bi = tg.InF("b", 40); return ([ai, bi], [tg.B.Emit1("Mod", [ai, bi], GeneralKit.A(("fmod", true)))]); }, a, b); }
        { var a = new float[] { 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 2.4f, 2.6f, 3.5f, -3.5f, 0f, 100.5f }; Run("round", tg => { var ai = tg.InF("a", 12); return ([ai], [tg.B.Emit1("Round", [ai])]); }, a); }
        {
            var x = Kit.Rand(4 * 5 * 6, 206);
            var w = Kit.Rand(4 * 2 * 9, 207, 0.5f);
            var b = Kit.Rand(4, 208);
            Save(dir, "convt", "in1", w);
            Save(dir, "convt", "in2", b);
            Run("convt", tg => { var xi = tg.InF("x", 1, 4, 5, 6); var wi = tg.WF("w", w, 4, 2, 3, 3); var bi = tg.WF("b", b, 4);
                return ([xi], [tg.B.Emit1("ConvTranspose", [xi, wi, bi], GeneralKit.A(("groups", 2L), ("strides", new long[] { 2, 2 }), ("pads", new long[] { 1, 1, 1, 1 }), ("output_padding", new long[] { 1, 1 })))]);
                }, x);
        }
        {
            var d = Kit.Rand(3 * 4 * 5, 209);
            int[] idx = [0, 2, 1, 3, 3, 0, 1, 2, 2, 1, 0, 3]; // batch_dims=1: [3, 2, 2] -> indexes dims 1,2
            int[] idx2 = [0, 1, 3, 2, 1, 0, 3, 4, 2, 2, 0, 0];
            Run("gathernd_b1", tg => { var di = tg.InF("d", 3, 4, 5); var ii = tg.In("i", DataType.I32, 3, 2, 2); return ([di, ii], [tg.B.Emit1("GatherND", [di, ii], GeneralKit.A(("batch_dims", 1L)))]); }, d, idx2);
        }
    }
}
