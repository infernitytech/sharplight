using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// A LoHa difference (w1a · w1b) ⊙ (w2a · w2b) (w·a [out, r], w·b [r, in · kh · kw]); with LyCORIS' Tucker cores (<c>hada_t1</c>,
/// <c>hada_t2</c> [r, r, kh, kw]) each factor is <c>einsum('i j k l, j r, i p -> p r k l', t, w·b, w·a)</c> with w·a [r, out], w·b [r, in].
/// </summary>
internal sealed class HadamardDelta(WeightInfo w1a, WeightInfo w1b, WeightInfo w2a, WeightInfo w2b, WeightInfo? t1, WeightInfo? t2) : LoraDelta
{
    /// <inheritdoc/>
    public override long Rows => t1 is null ? w1a.Shape[0].Fixed : w1a.Shape[1].Fixed;

    /// <inheritdoc/>
    public override long Columns => t1 is null ? Inner(w1b) : w1b.Shape[1].Fixed * Kernel(t1);

    /// <inheritdoc/>
    public override long? Rank => w1b.Shape[0].Fixed;

    /// <inheritdoc/>
    public override string Kind => "LoHa";

    /// <inheritdoc/>
    public override float[] Compute()
    {
        float[] first = Factor(w1a, w1b, t1);
        float[] second = Factor(w2a, w2b, t2);
        if (first.Length != second.Length)
        {
            throw new NotSupportedException($"LoHa factors {w1a.Shape} · {w1b.Shape} and {w2a.Shape} · {w2b.Shape} differ in size.");
        }

        for (int i = 0; i < first.Length; i++)
        {
            first[i] *= second[i];
        }

        return first;
    }

    private float[] Factor(WeightInfo a, WeightInfo b, WeightInfo? core)
    {
        long rank = b.Shape[0].Fixed;
        return core is null
            ? Multiply(DiffusionLoraPatch.Decode(a), DiffusionLoraPatch.Decode(b), Rows, rank, Columns)
            : Tucker(DiffusionLoraPatch.Decode(core), DiffusionLoraPatch.Decode(b), DiffusionLoraPatch.Decode(a), rank, Rows, b.Shape[1].Fixed, Kernel(core));
    }
}
