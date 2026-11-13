using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

public class DataMovementTests
{
    [Fact]
    public void WhereBroadcastF32()
    {
        int N = 3;
        int C = 4;
        int W = 6;
        var x = Kit.Rand(N * C * W, 101);
        var y = Kit.Rand(C * W, 102);
        var cond = Kit.Rand(C * 1, 103).Select(v => v > 0 ? 1f : 0f).ToArray();
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, W));
        var mask = tg.B.Emit1("Greater", [xin, tg.WF("z", [0.1f], 1, 1)]);
        tg.B.Output("w1", tg.B.Emit1("Where", [mask, xin, tg.WF("y", y, C, W)]));
        tg.B.Output("w2", tg.B.Emit1("Where", [tg.WF("c", cond, C, 1), xin, tg.WF("s", [-7f], 1)]));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
        Kit.AssertClose(x.Select((v, i) => v > 0.1f ? v : y[i % (C * W)]).ToArray(), Kit.Out(m, "w1", p), 0, 0);
        Kit.AssertClose(x.Select((v, i) => cond[i / W % C] != 0 ? v : -7f).ToArray(), Kit.Out(m, "w2", p), 0, 0);
    }

    [Fact]
    public void WhereOnIntegerTensors()
    {
        int T = 3;
        int n = 5;
        var a = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        var cond = new float[] { 1, 0, 1, 0, 1 };
        var b64 = Enumerable.Range(0, T * n).Select(i => -(long)i * 5_000_000_000L).ToArray();
                var tg2 = new TestGraph();
                var a32 = tg2.In("a", DataType.I32, TestGraph.T, n);
                var o32 = tg2.In("o", DataType.I32, TestGraph.T, n);
                var c2 = tg2.InF("c", n);
        var a64 = tg2.In("a64", DataType.I64, TestGraph.T, n);
        var o64 = tg2.In("o64", DataType.I64, TestGraph.T, n);
        tg2.B.Output("w32", tg2.B.Emit1("Where", [c2, a32, o32]));
        tg2.B.Output("w64", tg2.B.Emit1("Where", [c2, a64, o64]));
        using var m = Kit.Compile(tg2.B.Build());
        var o = a.Select(v => -v).ToArray();
        var l64 = a.Select(v => (long)v * 7_000_000_000L).ToArray();
        Kit.Set(m, a32, a);
        Kit.Set(m, o32, o);
        Kit.Set(m, c2, cond);
        GeneralKit.SetI64(m, a64, l64);
        GeneralKit.SetI64(m, o64, b64);
        var p = Kit.P(T);
        m.Run(p);
        Assert.Equal(a.Select((v, i) => cond[i % n] != 0 ? v : o[i]).ToArray(), Kit.OutI(m, "w32", p));
        Assert.Equal(l64.Select((v, i) => cond[i % n] != 0 ? v : b64[i]).ToArray(), GeneralKit.OutL(m, "w64", p));
    }

    [Fact]
    public void ExpandAndTile()
    {
        int N = 3;
        int C = 4;
        int W = 5;
        var x = Kit.Rand(N * C * 1, 104);
        var small = Kit.Rand(C * W, 105);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, 1));
        tg.B.Output("e", tg.B.Emit1("Expand", [xin], GeneralKit.A(("shape", new long[] { 1, C, W }))));
        var sv = tg.WF("s", small, 1, C, W);
        tg.B.Output("e2", tg.B.Emit1("Expand", [sv], GeneralKit.A(("shape", new long[] { 2, 3, C, W }))));
        var xt = tg.InF("xt", SpatialKit.DSym(2, 3));
        tg.B.Output("t", tg.B.Emit1("Tile", [xt], GeneralKit.A(("repeats", new long[] { 1, 3, 2 }))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var xtv = Kit.Rand(N * 6, 106);
        Kit.Set(m, xt, xtv);
        var p = Kit.P(N);
        m.Run(p);
        var e = new float[N * C * W];
        for (int i = 0; i < e.Length; i++)
        {
            e[i] = x[i / W];
        }

        Kit.AssertClose(e, Kit.Out(m, "e", p), 0, 0);
        var e2 = new float[2 * 3 * C * W];
        for (int i = 0; i < e2.Length; i++)
        {
            e2[i] = small[i % (C * W)];
        }

        Kit.AssertClose(e2, Kit.Out(m, "e2", p), 0, 0);
        var t = new float[N * 6 * 6];
        for (int n = 0; n < N; n++)
        {
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    t[(n * 6 + i) * 6 + j] = xtv[n * 6 + (i % 2) * 3 + j % 3];
                }
            }
        }

        Kit.AssertClose(t, Kit.Out(m, "t", p), 0, 0);
    }

    [Fact]
    public void TileAndExpandIntegers()
    {
        var tg = new TestGraph();
        var xi = tg.In("x", DataType.I64, 2, 3);
        var yi = tg.In("y", DataType.I32, 1, 3);
        tg.B.Output("t", tg.B.Emit1("Tile", [xi], GeneralKit.A(("repeats", new long[] { 2, 2 }))));
        tg.B.Output("e", tg.B.Emit1("Expand", [yi], GeneralKit.A(("shape", new long[] { 4, 3 }))));
        using var m = Kit.Compile(tg.B.Build());
        long[] x = [1, 2, 3, 40_000_000_000, 5, 6];
        int[] y = [7, 8, 9];
        GeneralKit.SetI64(m, xi, x);
        Kit.Set(m, yi, y);
        var p = Kit.P(1);
        m.Run(p);
        var t = new long[4 * 6];
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                t[i * 6 + j] = x[(i % 2) * 3 + j % 3];
            }
        }

        Assert.Equal(t, GeneralKit.OutL(m, "t", p));
        Assert.Equal(Enumerable.Range(0, 12).Select(i => y[i % 3]).ToArray(), Kit.OutI(m, "e", p));
    }

    public static IEnumerable<object[]> GatherAxisCases()
    {
        foreach (var t in new[] { DataType.I32, DataType.I64 })
        {
            foreach (var (axis, dims, idxDims) in new (int, long[], long[])[] { (0, [5, 4], [3]), (1, [3, 6], [2, 2]), (-1, [2, 3, 4], [5]), (1, [2, 5, 3], [4]), (0, [6], [2, 3]) })
            {
                yield return [t == DataType.I64, axis, dims, idxDims];
            }
        }
    }

    [Theory, MemberData(nameof(GatherAxisCases))]
    public void GatherAxis(bool idx64, int axis, long[] dims, long[] idxDims)
    {
        var data = Kit.Rand((int)GeneralKit.P(dims), 107);
        int ax = axis < 0 ? axis + dims.Length : axis;
        long A = dims[ax];
        var rng = new Random(3);
        var idx = Enumerable.Range(0, (int)GeneralKit.P(idxDims)).Select(_ => (long)rng.Next((int)(2 * A)) - A).ToArray(); // includes negatives
        var tg = new TestGraph();
        var din = tg.InF("d", dims.Select(d => (Dim)d).ToArray());
        var iin = tg.In("i", idx64 ? DataType.I64 : DataType.I32, idxDims.Select(d => (Dim)d).ToArray());
        tg.B.Output("y", tg.B.Emit1("GatherAxis", [din, iin], GeneralKit.A(("axis", (long)axis))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, din, data);
        if (idx64)
        {
            GeneralKit.SetI64(m, iin, idx);
        }
        else
        {
            Kit.Set(m, iin, idx.Select(v => (int)v).ToArray());
        }

        var p = Kit.P(1);
        m.Run(p);
        long outer = GeneralKit.P(dims[..ax]);
        long inner = GeneralKit.P(dims[(ax + 1)..]);
        var exp = new List<float>();
        for (long o = 0; o < outer; o++)
        {
            foreach (var id in idx) { long ii = id < 0 ? id + A : id; for (long i = 0; i < inner; i++)
                {
                    exp.Add(data[(o * A + ii) * inner + i]);
                }
            }
        }

        Kit.AssertClose(exp.ToArray(), Kit.Out(m, "y", p), 0, 0);
    }

    [Fact]
    public void GatherAxisIntegerDataAndSymbolicIndices()
    {
        var tg = new TestGraph();
        var d = tg.In("d", DataType.I64, 6);
        var i = tg.In("i", DataType.I32, TestGraph.T);
        tg.B.Output("y", tg.B.Emit1("GatherAxis", [d, i], GeneralKit.A(("axis", 0L))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8);
        long[] data = [10, 20, 30, 40_000_000_000, 50, 60];
        GeneralKit.SetI64(m, d, data);
        Kit.Set(m, i, new[] { 3, -1, 0 });
        var p = Kit.P(3);
        m.Run(p);
        Assert.Equal(new long[] { 40_000_000_000, 60, 10 }, GeneralKit.OutL(m, "y", p));
    }

    [Fact]
    public void GatherElementsMatchesReference()
    {
        long[] dd = [3, 4, 5];
        long[] id = [3, 2, 5];
        int axis = 1;
        var data = Kit.Rand((int)GeneralKit.P(dd), 108);
        var rng = new Random(5);
        var idx = Enumerable.Range(0, (int)GeneralKit.P(id)).Select(_ => rng.Next(-4, 4)).ToArray();
        var tg = new TestGraph();
        var din = tg.InF("d", dd.Select(d => (Dim)d).ToArray());
        var iin = tg.In("i", DataType.I32, id.Select(d => (Dim)d).ToArray());
        tg.B.Output("y", tg.B.Emit1("GatherElements", [din, iin], GeneralKit.A(("axis", (long)axis))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, din, data);
        Kit.Set(m, iin, idx);
        var p = Kit.P(1);
        m.Run(p);
        var ds = GeneralKit.Strides(dd);
        var exp = new float[idx.Length];
        for (int l = 0; l < exp.Length; l++)
        {
            var ix = GeneralKit.Unravel(l, id);
            long k = idx[l] < 0 ? idx[l] + dd[axis] : idx[l];
            ix[axis] = k;
            exp[l] = data[ix.Zip(ds, (a, b) => a * b).Sum()];
        }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 0, 0);
    }

    [Theory, InlineData(0), InlineData(1)]
    public void GatherNd(int batchDims)
    {
        long[] dd = [3, 4, 5];
        long[] id = batchDims == 0 ? [2, 3, 2] : [3, 2, 2]; // batch_dims=1: idx [3, 2, q=2] indexes data dims 1..2
        var data = Kit.Rand((int)GeneralKit.P(dd), 109);
        var rng = new Random(6);
        var idx = new int[(int)GeneralKit.P(id)];
        for (int l = 0; l < idx.Length; l++)
        {
            idx[l] = rng.Next(-3, 3) switch { var v when v < 0 => v, var v => v };
        }

        var tg = new TestGraph();
        var din = tg.InF("d", dd.Select(d => (Dim)d).ToArray());
        var iin = tg.In("i", DataType.I32, id.Select(d => (Dim)d).ToArray());
        tg.B.Output("y", tg.B.Emit1("GatherND", [din, iin], GeneralKit.A(("batch_dims", (long)batchDims))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, din, data);
        Kit.Set(m, iin, idx);
        var p = Kit.P(1);
        m.Run(p);
        var ds = GeneralKit.Strides(dd);
        int q = (int)id[^1];
        long tuples = GeneralKit.P(id[..^1]);
        long slice = GeneralKit.P(dd[(batchDims + q)..]);
        long perBatch = GeneralKit.P(dd[batchDims..]);
        long tpb = GeneralKit.P(id[batchDims..^1]);
        var exp = new List<float>();
        for (long t = 0; t < tuples; t++)
        {
            long off = t / tpb * perBatch;
            for (int j = 0; j < q; j++)
            {
                long v = idx[t * q + j];
                if (v < 0)
                {
                    v += dd[batchDims + j];
                }
                off += v * ds[batchDims + j];
            }
            for (long i = 0; i < slice; i++)
            {
                exp.Add(data[off + i]);
            }
        }
        Kit.AssertClose(exp.ToArray(), Kit.Out(m, "y", p), 0, 0);
    }

    private static float Red(string r, float a, float b) => r switch { "add" => a + b, "mul" => a * b, "max" => Math.Max(a, b), "min" => Math.Min(a, b), _ => b };

    [Theory, InlineData("none"), InlineData("add"), InlineData("mul"), InlineData("max"), InlineData("min")]
    public void ScatterElements(string red)
    {
        long[] dd = [4, 5];
        long[] id = [3, 5];
        int axis = 0;
        var data = Kit.Rand(20, 110);
        var upd = Kit.Rand(15, 111);
        var idx = new int[15];
        for (int i = 0; i < 15; i++)
        {
            idx[i] = (i * 7 + 1) % 4 - (i % 5 == 0 ? 4 : 0); // duplicates and negatives
        }

        var tg = new TestGraph();
        var din = tg.InF("d", 4, 5);
        var iin = tg.In("i", DataType.I32, 3, 5);
        var uin = tg.InF("u", 3, 5);
        tg.B.Output("y", tg.B.Emit1("ScatterElements", [din, iin, uin], GeneralKit.A(("axis", (long)axis), ("reduction", red))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, din, data);
        Kit.Set(m, iin, idx);
        Kit.Set(m, uin, upd);
        var p = Kit.P(1);
        m.Run(p);
        var exp = (float[])data.Clone();
        for (int l = 0; l < 15; l++)
        {
            var ix = GeneralKit.Unravel(l, id);
            ix[axis] = idx[l] < 0 ? idx[l] + 4 : idx[l];
            long o = ix[0] * 5 + ix[1];
            exp[o] = Red(red, exp[o], upd[l]);
        }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-6, 1e-6, red);
    }

    [Theory, InlineData("none"), InlineData("add"), InlineData("max")]
    public void ScatterNd(string red)
    {
        var data = Kit.Rand(4 * 3 * 2, 112);
        var upd = Kit.Rand(3 * 2 * 2, 113);
        int[] idx = [0, 1, 3, 2, 0, 1];  // [3, 2]: rows (0,1), (3,2), (0,1) duplicate
        var tg = new TestGraph();
        var din = tg.InF("d", 4, 3, 2);
        var iin = tg.In("i", DataType.I32, 3, 2);
        var uin = tg.InF("u", 3, 2);
        tg.B.Output("y", tg.B.Emit1("ScatterND", [din, iin, uin], GeneralKit.A(("reduction", red))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, din, data);
        Kit.Set(m, iin, idx);
        Kit.Set(m, uin, upd.Take(6).ToArray());
        var p = Kit.P(1);
        m.Run(p);
        var exp = (float[])data.Clone();
        for (int t = 0; t < 3; t++)
        {
            for (int i = 0; i < 2; i++)
            {
                long o = (idx[t * 2] * 3 + idx[t * 2 + 1]) * 2 + i;
                exp[o] = Red(red, exp[o], upd[t * 2 + i]);
            }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-6, 1e-6, red);
    }

    [Theory, InlineData(false, false), InlineData(true, false), InlineData(false, true), InlineData(true, true)]
    public void CumSum(bool exclusive, bool reverse)
    {
        long[] dims = [3, 5, 4];
        foreach (int axis in new[] { 0, 1, 2 })
        {
            var x = Kit.Rand((int)GeneralKit.P(dims), 114);
            var tg = new TestGraph();
            var xin = tg.InF("x", dims.Select(d => (Dim)d).ToArray());
            tg.B.Output("y", tg.B.Emit1("CumSum", [xin], GeneralKit.A(("axis", (long)axis), ("exclusive", exclusive), ("reverse", reverse))));
            using var m = Kit.Compile(tg.B.Build());
            Kit.Set(m, xin, x);
            var p = Kit.P(1);
            m.Run(p);
            var ds = GeneralKit.Strides(dims);
            var exp = new float[x.Length];
            for (int l = 0; l < x.Length; l++)
            {
                var ix = GeneralKit.Unravel(l, dims);
                double s = 0;
                long n = dims[axis];
                for (long k = 0; k < n; k++)
                {
                    bool inc = reverse ? k >= ix[axis] : k <= ix[axis];
                    if (exclusive && k == ix[axis])
                    {
                        inc = false;
                    }

                    if (!inc)
                    {
                        continue;
                    }

                    ix[axis] = GeneralKit.Unravel(l, dims)[axis];
                    var jx = (long[])ix.Clone();
                    jx[axis] = k;
                    s += x[jx.Zip(ds, (a, b) => a * b).Sum()];
                }
                exp[l] = (float)s;
            }
            Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-5, 1e-5, $"axis {axis}");
        }
    }

    [Theory, InlineData("max", 1, true, false), InlineData("min", 0, false, false), InlineData("max", 2, true, true), InlineData("min", 1, true, true), InlineData("max", -1, false, true)]
    public void ArgReduce(string mode, int axis, bool keep, bool last)
    {
        long[] dims = [4, 6, 5];
        var x = Kit.Rand((int)GeneralKit.P(dims), 115).Select(v => (float)Math.Round(v * 3)).ToArray();  // many ties
        var tg = new TestGraph();
        var xin = tg.InF("x", dims.Select(d => (Dim)d).ToArray());
        tg.B.Output("y", tg.B.Emit1("ArgReduce", [xin], GeneralKit.A(("mode", mode), ("axis", (long)axis), ("keepdims", keep), ("select_last_index", last))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(1);
        m.Run(p);
        int ax = axis < 0 ? axis + 3 : axis;
        var ds = GeneralKit.Strides(dims);
        var od = dims.Where((_, i) => i != ax).ToArray();
        var exp = new int[GeneralKit.P(od)];
        for (int l = 0; l < exp.Length; l++)
        {
            var ix2 = GeneralKit.Unravel(l, od);
            var ix = new long[3];
            for (int d = 0, k = 0; d < 3; d++)
            {
                ix[d] = d == ax ? 0 : ix2[k++];
            }

            int best = 0;
            float bv = 0;
            for (int a = 0; a < dims[ax]; a++)
            {
                ix[ax] = a;
                float v = x[ix.Zip(ds, (u, w) => u * w).Sum()];
                if (a == 0)
                {
                    bv = v;
                    continue;
                }
                bool better = mode == "max" ? (last ? v >= bv : v > bv) : (last ? v <= bv : v < bv);
                if (better)
                {
                    bv = v;
                    best = a;
                }
            }
            exp[l] = best;
        }
        Assert.Equal(exp, Kit.OutI(m, "y", p));
        Assert.Equal(keep ? 3 : 2, m.Graph.Outputs["y"].Shape.Rank);
    }

    [Theory, InlineData(true, 3, -1), InlineData(false, 3, -1), InlineData(true, 6, -1), InlineData(true, 2, 0), InlineData(false, 4, 1)]
    public void TopK(bool largest, int k, int axis)
    {
        long[] dims = [5, 6, 7];
        var x = Kit.Rand((int)GeneralKit.P(dims), 116).Select(v => (float)Math.Round(v * 4)).ToArray();  // ties
        int ax = axis < 0 ? axis + 3 : axis;
        if (k > dims[ax])
        {
            k = (int)dims[ax];
        }

        var tg = new TestGraph();
        var xin = tg.InF("x", dims.Select(d => (Dim)d).ToArray());
        var outs = tg.B.Emit("TopK", [xin], GeneralKit.A(("k", (long)k), ("axis", (long)axis), ("largest", largest)));
        tg.B.Output("v", outs[0]);
        tg.B.Output("i", outs[1]);
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(1);
        m.Run(p);
        var ds = GeneralKit.Strides(dims);
        var od = (long[])dims.Clone();
        od[ax] = k;
        var ev = new float[GeneralKit.P(od)];
        var ei = new int[ev.Length];
        var os = GeneralKit.Strides(od);
        for (long l = 0; l < GeneralKit.P(dims) / dims[ax]; l++)
        {
            var rest = dims.Where((_, i) => i != ax).ToArray();
            var r2 = GeneralKit.Unravel(l, rest);
            var ix = new long[3];
            for (int d = 0, q = 0; d < 3; d++)
            {
                ix[d] = d == ax ? 0 : r2[q++];
            }

            var order = Enumerable.Range(0, (int)dims[ax]).Select(a => { ix[ax] = a; return (a, v: x[ix.Zip(ds, (u, w) => u * w).Sum()]); })
                .OrderBy(t => largest ? -t.v : t.v).ThenBy(t => t.a).Take(k).ToArray();
            for (int j = 0; j < k; j++)
            {
                ix[ax] = j;
                long o = ix.Zip(os, (u, w) => u * w).Sum();
                ev[o] = order[j].v;
                ei[o] = order[j].a;
            }
        }
        Kit.AssertClose(ev, Kit.Out(m, "v", p), 0, 0);
        Assert.Equal(ei, Kit.OutI(m, "i", p));
    }

    [Theory, InlineData(true, 0), InlineData(false, 0), InlineData(true, 1), InlineData(false, -1), InlineData(true, 3)]
    public void Trilu(bool upper, int k)
    {
        var x = Kit.Rand(2 * 5 * 6, 117);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(5, 6));
        tg.B.Output("y", tg.B.Emit1("Trilu", [xin], GeneralKit.A(("upper", upper), ("k", (long)k))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        var p = Kit.P(2);
        m.Run(p);
        var exp = x.Select((v, l) => { int i = l / 6 % 5, j = l % 6; return (upper ? j - i >= k : j - i <= k) ? v : 0f; }).ToArray();
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 0, 0);
    }

    [Theory, InlineData(-1), InlineData(0), InlineData(1)]
    public void OneHot(int axis)
    {
        int[] idx = [2, 0, -1, 3, 7, 1]; // -1 wraps to depth-1; 7 is out of range -> all off
        var tg = new TestGraph();
        var iin = tg.In("i", DataType.I32, 2, 3);
        tg.B.Output("y", tg.B.Emit1("OneHot", [iin], GeneralKit.A(("depth", 4L), ("axis", (long)axis), ("on", 5f), ("off", -1f))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, iin, idx);
        var p = Kit.P(1);
        m.Run(p);
        int ax = axis < 0 ? axis + 3 : axis;
        long[] od = ax == 0 ? [4, 2, 3] : ax == 1 ? [2, 4, 3] : [2, 3, 4];
        var exp = new float[24];
        for (int l = 0; l < 24; l++)
        {
            var ix = GeneralKit.Unravel(l, od);
            int d = (int)ix[ax];
            var rest = ix.Where((_, i) => i != ax).ToArray();
            int v = idx[rest[0] * 3 + rest[1]];
            if (v < 0)
            {
                v += 4;
            }

            exp[l] = v == d ? 5f : -1f;
        }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 0, 0);
    }
}
