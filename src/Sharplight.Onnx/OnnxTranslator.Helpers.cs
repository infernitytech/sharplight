using System.Buffers.Binary;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Shared building blocks for the operator translators: input access, emission and layout helpers.
internal sealed partial class OnnxTranslator
{
    private Value X(TranslatedTensor?[] ins, int i, string? hint = null)
        => i < ins.Length && ins[i] is { } t ? AsValue(t, hint) : throw new InvalidDataException($"missing input {i}");

    /// <summary>A constant input; symbolic shape values (<see cref="SymbolicInts"/>) only where the caller handles them.</summary>
    private OnnxConst ConstIn(TranslatedTensor?[] ins, int i, string what, bool symbolic = false)
        => i < ins.Length && ins[i] is { } t
            ? NoSymbols(t.C ?? throw new NotSupportedException($"{what} must be a compile-time constant (initializer or constant-folded value)"), what, symbolic)
            : throw new InvalidDataException($"missing input {i} ({what})");

    private OnnxConst? OptConst(TranslatedTensor?[] ins, int i, string what, bool symbolic = false)
        => i < ins.Length && ins[i] is { } t ? NoSymbols(t.C ?? throw new NotSupportedException($"{what} must be a compile-time constant"), what, symbolic) : null;

    private static OnnxConst NoSymbols(OnnxConst c, string what, bool allowed)
        => allowed || !c.HasSymbols
            ? c
            : throw new NotSupportedException($"{what} depends on a symbolic size ({string.Join(", ", c.Ints().Where(SymbolicInts.IsSymbolic).Select(SymbolicInts.Format).Distinct())})");

    private TranslatedTensor? Opt(TranslatedTensor?[] ins, int i) => i < ins.Length ? ins[i] : null;

    /// <summary>Builds a long[] from mixed int/long arguments.</summary>
    private static long[] L(params long[] v) => v;

    private Value Emit(string op, Value[] inputs, Attributes? attrs = null, string? name = null)
    {
        // A leading 0 in a reshape target means "keep the symbolic batch dim"; for a fixed outer dim spell it out.
        if (op == "Reshape" && attrs is not null && !Sym(inputs[0]) && inputs[0].Rank > 0 && attrs.Get<long[]>("shape") is { Length: > 0 } sh && sh[0] == 0)
        {
            sh = (long[])sh.Clone();
            sh[0] = inputs[0].Shape[0].Fixed;
            attrs = Attributes.From(("shape", sh));
        }
        return _b.Emit1(op, inputs, attrs, name);
    }

    private Value Rs(Value x, params long[] shape) => Emit("Reshape", [x], Attributes.From(("shape", shape)));

    /// <summary>Transpose that is a no-op for the identity permutation and refuses to move the symbolic batch axis.</summary>
    private Value Permute(Value x, int[] perm)
    {
        if (perm.SequenceEqual(Enumerable.Range(0, perm.Length)))
        {
            return x;
        }

        if (Sym(x) && perm[0] != 0)
        {
            throw new NotSupportedException("permuting the symbolic batch axis away from the outermost position is not supported");
        }

        return _b.Permute(x, perm);
    }

    /// <summary>Index operand as an i32 IR value (dynamic indices are cast, constants become a Constant).</summary>
    private Value Idx(TranslatedTensor t)
    {
        if (t.V is { } v)
        {
            return v.Type == DataType.I32 ? v : t.CachedIndex ??= _b.Emit1("Cast", [v], Attributes.From(("type", "i32")));
        }

        var c = t.C!;
        if (c.HasSymbols)
        {
            return t.CachedValue ??= SymbolicData(c, DataType.I32, c.Name != "" ? c.Name : null);
        }

        var dims = c.Rank == 0 ? [1L] : c.Dims;
        var ints = c.Ints();
        var bytes = new byte[ints.Length * 4];
        for (int i = 0; i < ints.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4), (int)ints[i]);
        }

        return _b.Constant(c.Name != "" ? c.Name : "idx", DataType.I32, new Shape(dims.Select(d => (Dim)d).ToArray()), bytes);
    }

    /// <summary>Dims of <paramref name="x"/> with 0 standing for the symbolic batch dimension (the IR Reshape convention).</summary>
    private static long[] Fixed(Value x) => x.Shape.Dims.Select(d => d.IsSymbolic ? 0L : d.Fixed).ToArray();

    /// <summary>Concat of any number of values (the IR op takes at most 64).</summary>
    private Value ConcatMany(int axis, Value[] vals)
    {
        var cur = vals;
        while (cur.Length > 64)
        {
            cur = [.. cur.Chunk(64).Select(c => c.Length == 1 ? c[0] : _b.Concat(axis, c))];
        }

        return cur.Length == 1 ? cur[0] : _b.Concat(axis, cur);
    }

    private static byte[] ToBytes(float[] f)
    {
        var b = new byte[f.Length * 4];
        Buffer.BlockCopy(f, 0, b, 0, b.Length);
        return b;
    }

    /// <summary>Resolves a possibly negative ONNX axis against <paramref name="rank"/>.</summary>
    private static int NormalizeAxis(long axis, int rank) => OnnxFold.NormalizeAxis(axis, rank);
}
