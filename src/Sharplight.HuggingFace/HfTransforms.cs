using System.Buffers.Binary;
using System.Numerics;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Reusable weight transforms for architecture mappings. Anything that does not change bytes is a
/// zero-copy view (<see cref="WeightInfo.As"/>); everything else is lazily computed through
/// <see cref="WeightInfo.Derived"/> and only runs when the weight is first read.
/// </summary>
public static class HfTransforms
{
    /// <summary>Zero-copy rename. 1-D tensors that are not F32 are widened to F32 (llama.cpp keeps all 1-D tensors in F32).</summary>
    public static WeightInfo Plain(WeightInfo src, string name)
        => src.Shape.Rank <= 1 && src.Type != DataType.F32 ? ToF32(src, name) : src.As(name);

    /// <summary>Lazily converts any float/int tensor to F32, optionally reshaping and applying an element-wise function.</summary>
    public static WeightInfo ToF32(WeightInfo src, string name, Shape? shape = null, Func<float, float>? map = null)
    {
        var s = shape ?? src.Shape;
        if (s.Elements() != src.Shape.Elements()) throw new ArgumentException($"{src.Name}: cannot view {src.Shape} as {s}.");
        return WeightInfo.Derived(name, DataType.F32, s, () => EncodeF32(DecodeF32(src, map)));
    }

    /// <summary>
    /// RMSNorm weights: HF's Qwen3-Next family computes <c>x * (1 + w)</c>; the canonical form folds the
    /// 1 into the stored weight (llama.cpp: <c>data + 1</c> after converting to f32). Result is F32.
    /// </summary>
    public static WeightInfo NormPlusOne(WeightInfo src, string name) => ToF32(src, name, map: static v => v + 1f);

    /// <summary>
    /// Reorders V heads from grouped-by-K-head order to tiled order (llama.cpp <c>_reorder_v_heads</c>):
    /// the axis is viewed as <c>[numK, vPerK, headDim]</c> and permuted to <c>[vPerK, numK, headDim]</c>.
    /// </summary>
    /// <param name="data">The tensor bytes, row-major.</param>
    /// <param name="outerElems">Product of dimensions before the reordered axis.</param>
    /// <param name="innerElems">Product of dimensions after the reordered axis, times headDim.</param>
    public static byte[] ReorderVHeads(byte[] data, int elemBytes, long outerElems, int numK, int vPerK, long innerElems)
    {
        long block = innerElems * elemBytes;
        long axisBytes = block * numK * vPerK;
        if (data.LongLength != outerElems * axisBytes)
            throw new ArgumentException($"Reorder: {data.LongLength} bytes, expected {outerElems * axisBytes}.");
        var dst = new byte[data.Length];
        for (long o = 0; o < outerElems; o++)
            for (int k = 0; k < numK; k++)
                for (int v = 0; v < vPerK; v++)
                    Buffer.BlockCopy(data, checked((int)(o * axisBytes + (k * vPerK + v) * block)),
                                     dst, checked((int)(o * axisBytes + (v * numK + k) * block)), checked((int)block));
        return dst;
    }

    /// <summary>Decodes a weight to F32 values (F32/F16/BF16/F64 and integer types), optionally mapping each element.</summary>
    public static float[] DecodeF32(WeightInfo w, Func<float, float>? map = null)
    {
        var bytes = w.ReadAll();
        long n = w.Shape.Elements();
        var f = new float[n];
        var t = w.Type;
        if (t == DataType.F32) System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bytes).CopyTo(f);
        else if (t == DataType.BF16)
            for (int i = 0; i < f.Length; i++) f[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(i * 2)) << 16);
        else if (t == DataType.F16)
            for (int i = 0; i < f.Length; i++) f[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(bytes.AsSpan(i * 2));
        else if (t == DataType.F64)
            for (int i = 0; i < f.Length; i++) f[i] = (float)BinaryPrimitives.ReadDoubleLittleEndian(bytes.AsSpan(i * 8));
        else if (t == DataType.I8) for (int i = 0; i < f.Length; i++) f[i] = (sbyte)bytes[i];
        else if (t == DataType.U8) for (int i = 0; i < f.Length; i++) f[i] = bytes[i];
        else if (t == DataType.I16) for (int i = 0; i < f.Length; i++) f[i] = BinaryPrimitives.ReadInt16LittleEndian(bytes.AsSpan(i * 2));
        else if (t == DataType.I32) for (int i = 0; i < f.Length; i++) f[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * 4));
        else if (t == DataType.U32) for (int i = 0; i < f.Length; i++) f[i] = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(i * 4));
        else if (t == DataType.I64) for (int i = 0; i < f.Length; i++) f[i] = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(i * 8));
        else throw new NotSupportedException($"{w.Name}: cannot convert {t} to f32.");
        if (map != null) for (int i = 0; i < f.Length; i++) f[i] = map(f[i]);
        return f;
    }

    public static byte[] EncodeF32(float[] values)
    {
        var b = new byte[values.Length * 4];
        System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()).CopyTo(b);
        return b;
    }
}
