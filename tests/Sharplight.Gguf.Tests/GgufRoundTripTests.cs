using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Gguf.Tests;

public class GgufRoundTripTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "gguf-tests-" + Guid.NewGuid().ToString("N"));

    public GgufRoundTripTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_dir, true);
        }
        catch (IOException)
        {
            // Best-effort cleanup of the temp directory.
        }
    }

    private static byte[] Pattern(int length, int seed = 1)
    {
        var bytes = new byte[length];
        for (int i = 0; i < length; i++)
        {
            bytes[i] = (byte)(i * 7 + seed);
        }

        return bytes;
    }

    [Fact]
    public void Scalars_Arrays_Nested_RoundTrip()
    {
        var path = Path.Combine(_dir, "a.gguf");
        new GgufWriter()
            .AddMetadata("general.architecture", "toy")
            .AddMetadata("u8", (byte)200).AddMetadata("i8", (sbyte)-5)
            .AddMetadata("u16", (ushort)65000).AddMetadata("i16", (short)-30000)
            .AddMetadata("u32", 4000000000u).AddMetadata("i32", -2000000000)
            .AddMetadata("f32", 1.5f).AddMetadata("bool", true)
            .AddMetadata("u64", ulong.MaxValue).AddMetadata("i64", long.MinValue)
            .AddMetadata("f64", Math.PI).AddMetadata("str", "héllo wörld")
            .AddMetadata("toy.block_count", 12u)
            .AddMetadata("strs", new[] { "a", "", "ünï" })
            .AddMetadata("ints", new[] { 1, -2, 3 })
            .AddMetadata("uints", new[] { 1u, 2u })
            .AddMetadata("floats", new[] { 0.5f, 1.5f })
            .AddMetadata("bools", new[] { true, false, true })
            .AddMetadata("bytes", new byte[] { 1, 2, 3 })
            .AddMetadata("nested", new object[] { new[] { 1, 2 }, new[] { "x" }, new object[] { new[] { 1.0f } } })
            .Write(path);

        using var f = GgufFile.Open(path);
        Assert.Equal(3u, f.Version);
        Assert.Equal("toy", f.Architecture);
        Assert.Equal((byte)200, f.Metadata["u8"]);
        Assert.Equal((sbyte)-5, f.Metadata["i8"]);
        Assert.Equal((ushort)65000, f.Metadata["u16"]);
        Assert.Equal((short)-30000, f.Metadata["i16"]);
        Assert.Equal(4000000000u, f.Metadata["u32"]);
        Assert.Equal(-2000000000, f.Metadata["i32"]);
        Assert.Equal(1.5f, f.Metadata["f32"]);
        Assert.Equal(true, f.Metadata["bool"]);
        Assert.Equal(ulong.MaxValue, f.Metadata["u64"]);
        Assert.Equal(long.MinValue, f.Metadata["i64"]);
        Assert.Equal(Math.PI, f.Metadata["f64"]);
        Assert.Equal("héllo wörld", f.GetString("str"));
        Assert.Equal(new[] { "a", "", "ünï" }, f.GetArray<string>("strs"));
        Assert.Equal(new[] { 1, -2, 3 }, f.GetArray<int>("ints"));
        Assert.Equal(new[] { 1u, 2u }, f.GetArray<uint>("uints"));
        Assert.Equal(new[] { 0.5f, 1.5f }, f.GetArray<float>("floats"));
        Assert.Equal(new[] { true, false, true }, f.GetArray<bool>("bools"));
        Assert.Equal(new byte[] { 1, 2, 3 }, f.GetArray<byte>("bytes"));
        var nested = Assert.IsType<object[]>(f.Metadata["nested"]);
        Assert.Equal(new[] { 1, 2 }, Assert.IsType<int[]>(nested[0]));
        Assert.Equal(new[] { "x" }, Assert.IsType<string[]>(nested[1]));
        var inner = Assert.IsType<object[]>(nested[2]);
        Assert.Equal(new[] { 1.0f }, Assert.IsType<float[]>(inner[0]));

        // lenient helpers
        Assert.Equal(200u, f.GetUInt32("u8"));
        Assert.Equal(-2000000000L, f.GetInt64("i32"));
        Assert.Equal(1.5, f.GetFloat("f32"));
        Assert.Equal(12u, f.GetArch<uint>("block_count"));
        Assert.Equal(12, f.GetArch<int>("block_count"));
        Assert.Equal(7u, f.GetUInt32("missing", 7));
        Assert.False(f.TryGet<string>("u8", out _));
        Assert.True(f.TryGet<string>("str", out var s) && s == "héllo wörld");
        Assert.Equal(new long[] { 1, -2, 3 }, f.GetArray<long>("ints"));
    }

    [Fact]
    public void Tensors_ShapesSizesAndSources()
    {
        var path = Path.Combine(_dir, "t.gguf");
        var f32 = Pattern(4 * 3 * 4);               // 3 rows x 4 cols? dims [3,4]
        var q8 = Pattern(2 * 3 * 34, 3);            // dims [2,96] -> 2*3 blocks
        var q4k = Pattern(256 / 256 * 144 * 5, 9);  // dims [5,256]
        var f16 = Pattern(7 * 2, 11);               // dims [7]
        new GgufWriter()
            .AddMetadata("general.architecture", "toy")
            .AddTensor("w.f32", DataType.F32, f32, 3, 4)
            .AddTensor("w.q8", DataType.Q8_0, q8, 2, 96)
            .AddTensor("w.q4k", DataType.Q4_K, q4k, 5, 256)
            .AddTensor("v.f16", DataType.F16, f16, 7)
            .Write(path);

        using var f = GgufFile.Open(path);
        Assert.Equal(4, f.Tensors.Count);
        Assert.Equal(0, f.DataOffset % 32);
        foreach (var t in f.Tensors)
        {
            Assert.Equal(0, t.Offset % 32);
        }

        var t0 = f.GetTensor("w.f32");
        Assert.Equal(new long[] { 4, 3 }, t0.Ne);
        Assert.Equal(new Shape(3, 4), t0.Shape);
        Assert.Equal(DataType.F32, t0.DataType);
        Assert.Equal(0, t0.GgmlType);
        Assert.Equal(48, t0.ByteSize);

        var tq = f.GetTensor("w.q8");
        Assert.Equal(new Shape(2, 96), tq.Shape);
        Assert.Equal(6 * 34, tq.ByteSize);
        Assert.Equal(8, tq.GgmlType);
        Assert.Equal(720, f.GetTensor("w.q4k").ByteSize);
        Assert.Equal(new Shape(7), f.GetTensor("v.f16").Shape);
        Assert.Null(f.FindTensor("nope"));

        foreach (var (name, data) in new[] { ("w.f32", f32), ("w.q8", q8), ("w.q4k", q4k), ("v.f16", f16) })
        {
            var info = f.GetTensor(name);
            var w = info.ToWeightInfo();
            Assert.Equal(name, w.Name);
            Assert.Equal(info.Shape, w.Shape);
            Assert.Equal(data.Length, w.Source.ByteLength);
            Assert.Equal(data.Length, w.ByteSize);
            var buf = new byte[data.Length];
            w.Source.Read(0, buf);
            Assert.Equal(data, buf);
            // partial read
            var part = new byte[5];
            w.Source.Read(3, part);
            Assert.Equal(data.AsSpan(3, 5).ToArray(), part);
            unsafe
            {
                var p = w.Source.TryGetPointer();
                Assert.NotEqual(0, p);
                Assert.True(new ReadOnlySpan<byte>((void*)p, data.Length).SequenceEqual(data));
            }
            Assert.Throws<ArgumentOutOfRangeException>(() => w.Source.Read(data.Length - 1, new byte[2]));
        }
    }

    [Fact]
    public void CustomAlignment_IsHonored()
    {
        var path = Path.Combine(_dir, "al.gguf");
        var d1 = Pattern(16);
        var d2 = Pattern(16, 5);
        new GgufWriter { Alignment = 128 }
            .AddMetadata("general.architecture", "toy")
            .AddTensor("a", DataType.F32, d1, 4)
            .AddTensor("b", DataType.F32, d2, 4)
            .Write(path);
        using var f = GgufFile.Open(path);
        Assert.Equal(128, f.Alignment);
        Assert.Equal(0, f.DataOffset % 128);
        Assert.Equal(128, f.GetTensor("b").RelativeOffset);
        var buf = new byte[16];
        f.GetTensor("b").ToWeightInfo().Source.Read(0, buf);
        Assert.Equal(d2, buf);
    }

    [Fact]
    public void AllGgmlTypes_HaveConsistentLayout()
    {
        int known = 0;
        for (int id = 0; id < 44; id++)
        {
            var dt = GgmlTypes.TryGetDataType(id);
            if (dt is null)
            {
                continue;
            }

            known++;
            Assert.Equal(id, GgmlTypes.GetGgmlType(dt));
        }
        Assert.Equal(44 - 8, known); // 4,5,31-33,36-38 removed
        Assert.Equal((256, 66), (GgmlTypes.IQ2_XXS.BlockSize, GgmlTypes.IQ2_XXS.BlockBytes));
        Assert.Equal((64, 36), (GgmlTypes.NVFP4.BlockSize, GgmlTypes.NVFP4.BlockBytes));
        Assert.Equal(DataType.Q6_K, GgmlTypes.GetDataType(14));
        Assert.Equal(DataType.BF16, GgmlTypes.GetDataType(30));
        Assert.Equal(DataType.MXFP4, GgmlTypes.GetDataType(39));
        Assert.Throws<NotSupportedException>(() => GgmlTypes.GetDataType(31));
    }

    [Fact]
    public void GroupBitFormats_HaveGgmlIds()
    {
        Assert.Equal(DataType.Q1_0, GgmlTypes.GetDataType(41));
        Assert.Equal(DataType.Q2_0, GgmlTypes.GetDataType(42));
        Assert.Equal(DataType.Q2_0_G128, GgmlTypes.GetDataType(142));   // PrismML fork: PQ2_0
        Assert.Equal(DataType.Tern1_G128, GgmlTypes.GetDataType(143));  // PrismML fork: PTQ1_0
        Assert.Equal(142, GgmlTypes.GetGgmlType(DataType.Q2_0_G128));
        Assert.Equal(143, GgmlTypes.GetGgmlType(DataType.Tern1_G128));
        Assert.Equal((128, 18), (DataType.Q1_0.BlockSize, DataType.Q1_0.BlockBytes));
        Assert.Equal((64, 18), (DataType.Q2_0.BlockSize, DataType.Q2_0.BlockBytes));
        Assert.Equal((128, 34), (DataType.Q2_0_G128.BlockSize, DataType.Q2_0_G128.BlockBytes));
        Assert.Equal((128, 28), (DataType.Tern1_G128.BlockSize, DataType.Tern1_G128.BlockBytes));
    }

    [Fact]
    public void ManyTensorTypes_RoundTrip()
    {
        var path = Path.Combine(_dir, "many.gguf");
        var w = new GgufWriter().AddMetadata("general.architecture", "toy");
        var types = new[] { DataType.Q4_0, DataType.Q5_1, DataType.Q2_K, DataType.Q6_K, DataType.IQ4_XS, GgmlTypes.IQ3_S, GgmlTypes.TQ1_0, GgmlTypes.NVFP4, DataType.MXFP4, DataType.BF16, DataType.I32,
            DataType.Q1_0, DataType.Q2_0, DataType.Q2_0_G128, DataType.Tern1_G128 };
        foreach (var t in types)
        {
            w.AddTensor("t." + t.Name, t, Pattern((int)t.ByteSize(t.BlockSize * 2)), 2, t.BlockSize);
        }

        w.Write(path);
        using var f = GgufFile.Open(path);
        foreach (var t in types)
        {
            var info = f.GetTensor("t." + t.Name);
            Assert.Equal(t, info.DataType);
            Assert.Equal(new Shape(2, t.BlockSize), info.Shape);
            Assert.Equal(t.BlockBytes * 2, info.ByteSize);
        }
    }

    [Fact]
    public void Split_Files_AreMerged()
    {
        var p1 = Path.Combine(_dir, "m-00001-of-00002.gguf");
        var p2 = Path.Combine(_dir, "m-00002-of-00002.gguf");
        new GgufWriter().AddMetadata("general.architecture", "toy").AddMetadata("split.count", (ushort)2)
            .AddTensor("a", DataType.F32, Pattern(8), 2).Write(p1);
        new GgufWriter().AddMetadata("split.no", (ushort)1).AddTensor("b", DataType.F32, Pattern(8, 2), 2).Write(p2);
        using var m = GgufModelFiles.Open(p1);
        Assert.Equal(2, m.Files.Count);
        Assert.Equal(2, m.Tensors.Count);
        Assert.Equal("toy", m.Architecture);
        Assert.NotNull(m.FindTensor("b"));
        var buf = new byte[8];
        m.FindTensor("b")!.ToWeightInfo().Source.Read(0, buf);
        Assert.Equal(Pattern(8, 2), buf);
    }

    [Fact]
    public void BadMagic_And_Truncation_Throw()
    {
        var bad = Path.Combine(_dir, "bad.gguf");
        File.WriteAllBytes(bad, new byte[64]);
        Assert.Throws<InvalidDataException>(() => GgufFile.Open(bad));

        var good = Path.Combine(_dir, "g.gguf");
        new GgufWriter().AddMetadata("general.architecture", "toy").AddTensor("a", DataType.F32, Pattern(64), 16).Write(good);
        var bytes = File.ReadAllBytes(good);
        var trunc = Path.Combine(_dir, "trunc.gguf");
        File.WriteAllBytes(trunc, bytes[..(bytes.Length - 10)]);
        Assert.Throws<InvalidDataException>(() => GgufFile.Open(trunc));
    }

    [Fact]
    public void Dispose_InvalidatesSources()
    {
        var path = Path.Combine(_dir, "d.gguf");
        new GgufWriter().AddTensor("a", DataType.F32, Pattern(16), 4).Write(path);
        var f = GgufFile.Open(path);
        var src = f.GetTensor("a").ToWeightInfo().Source;
        f.Dispose();
        Assert.Throws<ObjectDisposedException>(() => src.Read(0, new byte[4]));
    }
}
