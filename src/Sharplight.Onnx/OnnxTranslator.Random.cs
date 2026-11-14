using System.Text;
using Sharplight.Graph;

namespace Sharplight.Onnx;

// Random number generation: RandomNormal/Uniform(-Like), Bernoulli and Multinomial on the IR Random and Multinomial ops. The generator is
// deterministic (Runtime.CounterRandom): a seeded node gives the same values on every provider and in every run. The seed is the bit pattern
// of the float "seed" attribute; an unseeded node uses the FNV-1a hash of its first output name, so it is equally reproducible and two
// unseeded nodes draw different streams.
internal sealed partial class OnnxTranslator
{
    private void RandomTensor(OnnxNode n, TranslatedTensor?[] ins)
    {
        bool normal = n.OpType.StartsWith("RandomNormal", StringComparison.Ordinal);
        var attributes = normal
            ? Attributes.From(("distribution", "normal"), ("mean", n.Float("mean", 0f)), ("scale", n.Float("scale", 1f)), ("seed", Seed(n)))
            : Attributes.From(("distribution", "uniform"), ("low", n.Float("low", 0f)), ("high", n.Float("high", 1f)), ("seed", Seed(n)));
        Value y;
        if (n.OpType.EndsWith("Like", StringComparison.Ordinal))
        {
            var like = ins[0]!;
            y = like.V is { } v
                ? Emit("Random", [v], attributes, Nm(n))
                : Emit("Random", [], attributes.With("shape", like.C!.Rank == 0 ? [1L] : like.C.Dims), Nm(n));
        }
        else
        {
            var shape = n.Ints("shape") ?? throw new InvalidDataException($"{n.OpType} needs a shape attribute");
            y = Emit("Random", [], attributes.With("shape", shape.Length == 0 ? [1L] : shape), Nm(n));
        }

        SetDyn(n, 0, ToElementType(y, (int)n.Int("dtype", OnnxType.Float), Nm(n)));
    }

    /// <summary>Bernoulli(p): 1 where a uniform draw is below p (the draw is the Random op over p's shape).</summary>
    private void Bernoulli(OnnxNode n, TranslatedTensor?[] ins)
    {
        var p = X(ins, 0);
        string name = Nm(n);
        var u = Emit("Random", [p], Attributes.From(("distribution", "uniform"), ("low", 0f), ("high", 1f), ("seed", Seed(n))), name + ".uniform");
        var y = Emit("Less", [u, p], null, name);
        SetDyn(n, 0, ToElementType(y, (int)n.Int("dtype", OnnxType.Float), name));
    }

    private void Multinomial(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var y = Emit("Multinomial", [x], Attributes.From(("sample_size", n.Int("sample_size", 1)), ("seed", Seed(n))), Nm(n));
        SetDyn(n, 0, y);
    }

    /// <summary>A float tensor as the IR carrier of an ONNX element type: i32 for integer types, f32 (bool as 0/1) otherwise.</summary>
    private Value ToElementType(Value y, int elemType, string name)
        => OnnxType.IsInteger(elemType) ? Emit("Cast", [y], Attributes.From(("type", "i32")), name + ".cast") : y;

    private static long Seed(OnnxNode n)
    {
        if (n.Has("seed"))
        {
            return BitConverter.SingleToInt32Bits(n.Float("seed", 0f));
        }

        return (long)Fnv1a(Encoding.UTF8.GetBytes(n.Outputs[0]));
    }

    private static ulong Fnv1a(byte[] bytes)
    {
        ulong hash = 0xCBF29CE484222325UL;
        foreach (var b in bytes)
        {
            hash = unchecked((hash ^ b) * 0x100000001B3UL);
        }

        return hash;
    }
}
