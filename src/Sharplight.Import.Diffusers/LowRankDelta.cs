using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// A plain LoRA / LoCon difference up · down (down [r, in (, kh, kw)], up [out, r (, 1, 1)]), with LyCORIS' optional Tucker core
/// (<c>lora_mid</c> [r, r, kh, kw] with a 1×1 down matrix), reconstructed as ComfyUI and LyCORIS do.
/// </summary>
internal sealed class LowRankDelta(WeightInfo down, WeightInfo up, WeightInfo? mid) : LoraDelta
{
    /// <summary>The down matrix.</summary>
    public WeightInfo Down => down;

    /// <summary>The up matrix.</summary>
    public WeightInfo Up => up;

    /// <inheritdoc/>
    public override long Rows => up.Shape[0].Fixed;

    /// <inheritdoc/>
    public override long Columns => mid is null ? Inner(down) : down.Shape[1].Fixed * Kernel(mid);

    /// <inheritdoc/>
    public override long? Rank => down.Shape[0].Fixed;

    /// <inheritdoc/>
    public override string Kind => mid is null ? "LoRA" : "LoCon (Tucker)";

    /// <inheritdoc/>
    public override float[] Compute()
    {
        long rank = down.Shape[0].Fixed;
        float[] downValues = DiffusionLoraPatch.Decode(down);
        if (mid is not null)
        {
            // down'[a, i, s] = Σ_b mid[a, b, s] · down[b, i]: the Tucker core spreads the 1×1 down matrix over the kernel.
            long inputs = down.Shape[1].Fixed;
            long kernel = Kernel(mid);
            float[] core = DiffusionLoraPatch.Decode(mid);
            var spread = new float[rank * inputs * kernel];
            for (long a = 0; a < rank; a++)
            {
                for (long b = 0; b < rank; b++)
                {
                    for (long i = 0; i < inputs; i++)
                    {
                        float d = downValues[b * inputs + i];
                        for (long s = 0; s < kernel; s++)
                        {
                            spread[(a * inputs + i) * kernel + s] += core[(a * rank + b) * kernel + s] * d;
                        }
                    }
                }
            }

            downValues = spread;
        }

        return Multiply(DiffusionLoraPatch.Decode(up), downValues, Rows, rank, Columns);
    }
}
