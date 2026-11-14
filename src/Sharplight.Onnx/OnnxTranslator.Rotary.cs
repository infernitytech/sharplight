using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// ai.onnx RotaryEmbedding (opset 23).
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// RotaryEmbedding: input [B, H, S, D] or [B, S, H·D] (num_heads), cos/sin caches [max position, R/2] indexed by position_ids [B, S]
    /// (or, without position ids, caches [B, S, R/2] per token), rotary_embedding_dim R (default D), interleaved pairs or two halves.
    /// Constant caches that are plain rotary tables (cos/sin of position · ωᵢ, times a constant magnitude) become the IR Rope op with
    /// per-frequency divisors; other caches are applied elementwise.
    /// </summary>
    private void OnnxRotaryEmbedding(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        bool fourD = x.Rank == 4;
        long heads = fourD ? x.Shape[1].Fixed : n.Int("num_heads", 0);
        if (!fourD && (x.Rank != 3 || heads <= 0))
        {
            throw new InvalidDataException("RotaryEmbedding needs a 4-D input or a 3-D input with num_heads");
        }

        long s = x.Shape[fourD ? 2 : 1].Fixed;
        long d = fourD ? x.Shape[3].Fixed : x.Shape[2].Fixed / heads;
        long rotary = n.Int("rotary_embedding_dim", 0) is > 0 and var r ? r : d;
        bool interleaved = n.Int("interleaved", 0) != 0;

        // [B, S, H, D] layout for the rotation.
        var xh = fourD ? Permute(x, [0, 2, 1, 3]) : Rs(x, 0, s, heads, d);
        var positions = Opt(ins, 3);
        Value y;
        if (positions is not null && RotaryFrequencies(ins[1], ins[2], rotary) is { } table && !Sym(x))
        {
            y = RopeRotation(xh, positions, table, rotary, interleaved, Nm(n));
        }
        else
        {
            Value cos, sin;
            if (positions is not null)
            {
                var ids = Idx(positions);
                cos = Emit("GatherAxis", [X(ins, 1), ids], Attributes.From(("axis", 0L)));
                sin = Emit("GatherAxis", [X(ins, 2), ids], Attributes.From(("axis", 0L)));
            }
            else
            {
                cos = X(ins, 1);
                sin = X(ins, 2);
            }

            long half = cos.Shape[cos.Rank - 1].Fixed;
            if (2 * half != rotary)
            {
                throw new InvalidDataException($"RotaryEmbedding caches of width {half} for {rotary} rotated dims");
            }

            y = Rotate(xh, Rs(cos, 0, s, 1, half), Rs(sin, 0, s, 1, half), interleaved);
        }

        SetDyn(n, 0, fourD
            ? Permute(y, [0, 2, 1, 3])
            : Emit("Reshape", [y], Attributes.From(("shape", L(0, s, heads * d))), Nm(n)));
    }

    /// <summary>The frequencies ω [R/2] and magnitude of constant caches with cache[p, i] = magnitude · cos/sin(p · ωᵢ), or null.</summary>
    private static (double[] Omega, double Magnitude)? RotaryFrequencies(TranslatedTensor? cosInput, TranslatedTensor? sinInput, long rotary)
    {
        if (cosInput?.C is not { Rank: 2 } cosCache || sinInput?.C is not { Rank: 2 } sinCache || cosCache.Dims[0] < 2 || cosCache.Dims[1] * 2 != rotary
            || !cosCache.Dims.SequenceEqual(sinCache.Dims))
        {
            return null;
        }

        var cos = cosCache.Floats();
        var sin = sinCache.Floats();
        int half = (int)cosCache.Dims[1];
        double magnitude = cos[0];
        if (magnitude <= 0)
        {
            return null;
        }

        var omega = new double[half];
        for (int i = 0; i < half; i++)
        {
            omega[i] = Math.Atan2(sin[half + i], cos[half + i]);
            if (omega[i] <= 0)
            {
                return null;
            }
        }

        const double Tolerance = 1e-4;
        for (long p = 0; p < cosCache.Dims[0]; p++)
        {
            for (int i = 0; i < half; i++)
            {
                double angle = p * omega[i];
                long k = p * half + i;
                if (Math.Abs(cos[k] - magnitude * Math.Cos(angle)) > Tolerance || Math.Abs(sin[k] - magnitude * Math.Sin(angle)) > Tolerance)
                {
                    return null;
                }
            }
        }

        return (omega, magnitude);
    }

    /// <summary>
    /// The IR Rope op on x [B, S, H, D] with positions [B, S]: tokens become rows, ωᵢ = base^(-2i/R) / factorᵢ with base 10000.
    /// </summary>
    private Value RopeRotation(Value x, TranslatedTensor positions, (double[] Omega, double Magnitude) table, long rotary, bool interleaved, string name)
    {
        const double Base = 10000;
        long b = x.Shape[0].Fixed;
        long s = x.Shape[1].Fixed;
        long h = x.Shape[2].Fixed;
        long d = x.Shape[3].Fixed;
        var factors = table.Omega.Select((w, i) => (float)(Math.Pow(Base, -2.0 * i / rotary) / w)).ToArray();
        var ids = Idx(positions);
        if (ids.Shape.Elements() != b * s)
        {
            // Positions [1, S] shared by every batch row.
            ids = Emit("Expand", [ids], Attributes.From(("shape", L(b, s))));
        }

        var rows = Rs(ids, b * s, 1);
        var rotated = Emit("Rope", [Rs(x, b * s, h, d), rows, _b.Constant(name + ".rope_factors", DataType.F32, new Shape(factors.Length), ToBytes(factors))],
            Attributes.From(("dims", rotary), ("base", (float)Base), ("style", interleaved ? "gptj" : "neox"), ("attn_factor", (float)table.Magnitude)));
        return Rs(rotated, b, s, h, d);
    }
}
