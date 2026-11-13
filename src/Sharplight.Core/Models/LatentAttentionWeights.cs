using Sharplight.Tensors;

namespace Sharplight.Models;

/// <summary>
/// Format-neutral weight layout conversions of multi-head latent attention (MLA) shared by importers. Checkpoints store the
/// key/value up-projection as one matrix <c>kv_up</c> [H·(nope + v), kv_lora] (per head: <c>nope</c> key rows, then <c>v</c>
/// value rows); the canonical absorbed form needs it per head as <c>attention.k_up.weight</c> [H, kv_lora, nope] (transposed,
/// maps a query's non-rotary part into latent space) and <c>attention.v_up.weight</c> [H, v, kv_lora] (maps the attended latent
/// back to the value head).
/// </summary>
public static class LatentAttentionWeights
{
    /// <summary>Splits a combined key/value up-projection into the canonical per-head key and value up-projections.</summary>
    /// <param name="kvUp">The combined up-projection [H·(nope + v), kv_lora] in any storage type.</param>
    /// <param name="heads">Attention heads H.</param>
    /// <param name="nopeDims">Non-rotary key dimensions per head.</param>
    /// <param name="valueDims">Value dimensions per head.</param>
    /// <param name="keyName">Name of the key up-projection.</param>
    /// <param name="valueName">Name of the value up-projection.</param>
    /// <param name="decode">Decodes the whole <paramref name="kvUp"/> to f32 (row-major [H·(nope + v), kv_lora]); only called for the key part.</param>
    /// <returns>The key up-projection (f16, transposed per head, decoded lazily) and the value up-projection (the value rows in
    /// the original storage type, gathered lazily).</returns>
    /// <exception cref="ArgumentException">The shape of <paramref name="kvUp"/> does not match the head layout.</exception>
    public static (WeightInfo KeyUp, WeightInfo ValueUp) SplitKeyValueUp(
        WeightInfo kvUp, int heads, int nopeDims, int valueDims, string keyName, string valueName, Func<WeightInfo, float[]> decode)
    {
        if (kvUp.Shape.Rank != 2 || kvUp.Shape[0].Fixed != (long)heads * (nopeDims + valueDims))
        {
            throw new ArgumentException($"{kvUp.Name}: expected [{heads}·({nopeDims} + {valueDims}), kv_lora], got {kvUp.Shape}.");
        }

        int latent = checked((int)kvUp.Shape[1].Fixed);
        int headRows = nopeDims + valueDims;
        var keyUp = WeightInfo.Derived(keyName, DataType.F16, new Shape(heads, latent, nopeDims), () =>
        {
            var w = decode(kvUp);
            var bytes = new byte[(long)heads * latent * nopeDims * 2];
            var halves = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Half>(bytes);
            for (int h = 0; h < heads; h++)
            {
                for (int n = 0; n < nopeDims; n++)
                {
                    long source = ((long)h * headRows + n) * latent;
                    for (int c = 0; c < latent; c++)
                    {
                        halves[(int)(((long)h * latent + c) * nopeDims + n)] = (Half)w[source + c];
                    }
                }
            }

            return bytes;
        });

        long rowBytes = kvUp.Type.ByteSize(latent);
        var valueUp = WeightInfo.Derived(valueName, kvUp.Type, new Shape(heads, valueDims, latent), () =>
        {
            var src = kvUp.ReadAll();
            var dst = new byte[(long)heads * valueDims * rowBytes];
            for (int h = 0; h < heads; h++)
            {
                long from = ((long)h * headRows + nopeDims) * rowBytes;
                Buffer.BlockCopy(src, (int)from, dst, (int)((long)h * valueDims * rowBytes), (int)(valueDims * rowBytes));
            }

            return dst;
        });
        return (keyUp, valueUp);
    }
}
