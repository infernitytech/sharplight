using System.Runtime.InteropServices;
using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

/// <summary>Writes Resize results to $SHARPLIGHT_ORT_DUMP (used by .work/venv onnxruntime cross-check scripts); no-op otherwise.</summary>
public class OrtDumpTests
{
    [Fact]
    public void DumpResize()
    {
        var dir = Environment.GetEnvironmentVariable("SHARPLIGHT_ORT_DUMP");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }

        Directory.CreateDirectory(dir);
        int N = 1;
        int C = 2;
        int H = 4;
        int W = 5;
        var x = Kit.Rand(N * C * H * W, 63);
        File.WriteAllBytes(Path.Combine(dir, "x.bin"), MemoryMarshal.AsBytes(x.AsSpan()).ToArray());
        foreach (var o in PadResizeTests.ResizeCases())
        {
            string mode = (string)o[0];
            string coord = (string)o[1];
            string nm = (string)o[2];
            int oh = (int)o[3];
            int ow = (int)o[4];
            var tg = new TestGraph();
            var xin = tg.InF("x", SpatialKit.D(N, C, H, W));
            tg.B.Output("y", tg.B.Emit1("Resize", [xin], Attributes.From(("size", new long[] { oh, ow }), ("mode", mode), ("coord", coord), ("nearest_mode", nm))));
            using var m = Kit.Compile(tg.B.Build());
            Kit.Set(m, xin, x);
            var p = Kit.P(1);
            m.Run(p);
            File.WriteAllBytes(Path.Combine(dir, $"{mode}_{coord}_{nm}_{oh}x{ow}.bin"), MemoryMarshal.AsBytes(Kit.Out(m, "y", p).AsSpan()).ToArray());
        }
    }
}
