namespace Sharplight.Cpu.Tests;

internal static class Quantizers
{
    public static byte[] Half(float f)
    {
        var h = (Half)f;
        return BitConverter.GetBytes(BitConverter.HalfToUInt16Bits(h));
    }
    public static float HalfV(byte[] b, int o) => (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(b, o));

    /// <summary>ggml quantize_row_q8_0_ref</summary>
    public static byte[] Q8_0(float[] x)
    {
        int nb = x.Length / 32;
        var o = new byte[nb * 34];
        for (int b = 0; b < nb; b++)
        {
            float amax = 0;
            for (int j = 0; j < 32; j++)
            {
                amax = MathF.Max(amax, MathF.Abs(x[b * 32 + j]));
            }

            float d = amax / 127f;
            float id = d != 0 ? 1f / d : 0;
            Half(d).CopyTo(o, b * 34);
            for (int j = 0; j < 32; j++)
            {
                o[b * 34 + 2 + j] = (byte)(sbyte)MathF.Round(x[b * 32 + j] * id, MidpointRounding.AwayFromZero);
            }
        }
        return o;
    }

    /// <summary>ggml quantize_row_q4_0_ref</summary>
    public static byte[] Q4_0(float[] x)
    {
        int nb = x.Length / 32;
        var o = new byte[nb * 18];
        for (int b = 0; b < nb; b++)
        {
            float amax = 0;
            float max = 0;
            for (int j = 0; j < 32; j++)
            {
                float v = x[b * 32 + j];
                if (MathF.Abs(v) > amax)
                {
                    amax = MathF.Abs(v);
                    max = v;
                }
            }
            float d = max / -8f;
            float id = d != 0 ? 1f / d : 0;
            Half(d).CopyTo(o, b * 18);
            for (int j = 0; j < 16; j++)
            {
                int x0 = Math.Min(15, (int)(x[b * 32 + j] * id + 8.5f));
                int x1 = Math.Min(15, (int)(x[b * 32 + j + 16] * id + 8.5f));
                o[b * 18 + 2 + j] = (byte)(x0 | (x1 << 4));
            }
        }
        return o;
    }

    public static float[] DeqQ8_0(byte[] q)
    {
        int nb = q.Length / 34;
        var r = new float[nb * 32];
        for (int b = 0; b < nb; b++) { float d = HalfV(q, b * 34); for (int j = 0; j < 32; j++)
            {
                r[b * 32 + j] = (sbyte)q[b * 34 + 2 + j] * d;
            }
        }
        return r;
    }

    public static float[] DeqQ4_0(byte[] q)
    {
        int nb = q.Length / 18;
        var r = new float[nb * 32];
        for (int b = 0; b < nb; b++)
        {
            float d = HalfV(q, b * 18);
            for (int j = 0; j < 16; j++)
            {
                byte v = q[b * 18 + 2 + j];
                r[b * 32 + j] = ((v & 0xF) - 8) * d;
                r[b * 32 + j + 16] = ((v >> 4) - 8) * d;
            }
        }
        return r;
    }
}
