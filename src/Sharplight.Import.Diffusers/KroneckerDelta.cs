using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// A LoKr difference kron(w1, w2): w1 [a, b] (stored, or <c>lokr_w1_a</c> · <c>lokr_w1_b</c>), w2 [c, d (, kh, kw)] (stored, <c>lokr_w2_a</c> ·
/// <c>lokr_w2_b</c>, or with the Tucker core <c>lokr_t2</c>), giving [a · c, b · d · kh · kw]; the alpha divides by the rank of the decomposed
/// factor (the second's when both are), and does not apply when both are stored whole (ComfyUI's and LyCORIS' convention).
/// </summary>
internal sealed class KroneckerDelta(WeightInfo? w1, WeightInfo? w1a, WeightInfo? w1b, WeightInfo? w2, WeightInfo? w2a, WeightInfo? w2b, WeightInfo? t2)
    : LoraDelta
{
    private long FirstRows => w1?.Shape[0].Fixed ?? w1a!.Shape[0].Fixed;

    private long FirstColumns => w1 is not null ? Inner(w1) : Inner(w1b!);

    private long SecondRows => w2?.Shape[0].Fixed ?? (t2 is null ? w2a!.Shape[0].Fixed : w2a!.Shape[1].Fixed);

    private long SecondColumns => w2 is not null ? Inner(w2) : t2 is null ? Inner(w2b!) : w2b!.Shape[1].Fixed * Kernel(t2);

    /// <inheritdoc/>
    public override long Rows => FirstRows * SecondRows;

    /// <inheritdoc/>
    public override long Columns => FirstColumns * SecondColumns;

    /// <inheritdoc/>
    public override long? Rank => w2 is null ? w2b!.Shape[0].Fixed : w1 is null ? w1b!.Shape[0].Fixed : null;

    /// <inheritdoc/>
    public override string Kind => "LoKr";

    /// <inheritdoc/>
    public override float[] Compute()
    {
        float[] first = w1 is not null
            ? DiffusionLoraPatch.Decode(w1)
            : Multiply(DiffusionLoraPatch.Decode(w1a!), DiffusionLoraPatch.Decode(w1b!), FirstRows, w1b!.Shape[0].Fixed, FirstColumns);
        float[] second = w2 is not null ? DiffusionLoraPatch.Decode(w2)
            : t2 is null ? Multiply(DiffusionLoraPatch.Decode(w2a!), DiffusionLoraPatch.Decode(w2b!), SecondRows, w2b!.Shape[0].Fixed, SecondColumns)
            : Tucker(DiffusionLoraPatch.Decode(t2), DiffusionLoraPatch.Decode(w2b!), DiffusionLoraPatch.Decode(w2a!), w2b!.Shape[0].Fixed, SecondRows,
                w2b.Shape[1].Fixed, Kernel(t2));
        // The kernel of a convolution's second factor stays innermost: out[i·c + k, (j·d + l)·s + q] = w1[i, j] · w2[k, l, q].
        long a = FirstRows;
        long b = FirstColumns;
        long c = SecondRows;
        long secondColumns = SecondColumns;
        var result = new float[Rows * Columns];
        for (long i = 0; i < a; i++)
        {
            for (long j = 0; j < b; j++)
            {
                float factor = first[i * b + j];
                for (long k = 0; k < c; k++)
                {
                    long row = i * c + k;
                    var destination = result.AsSpan((int)(row * Columns + j * secondColumns), (int)secondColumns);
                    var source = second.AsSpan((int)(k * secondColumns), (int)secondColumns);
                    for (int q = 0; q < destination.Length; q++)
                    {
                        destination[q] = factor * source[q];
                    }
                }
            }
        }

        return result;
    }
}
