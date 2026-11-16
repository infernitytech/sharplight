using Sharplight.Import.Gguf;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// 3-D convolution kernels of video VAEs in the layout of the frames-as-batch graphs: the kernel's time taps folded into its input
/// channels, so a 3-D convolution over a window of frames is one 2-D convolution over the frames stacked along the channels.
/// </summary>
internal static class Conv3dKernels
{
    /// <summary>
    /// A 3-D convolution kernel [O, C, kt, kh, kw] as the 2-D kernel [O, kt·C, kh, kw] of the stacked frame windows (time-tap-major input
    /// channels); quantized kernels are decoded to f32 first.
    /// </summary>
    public static WeightInfo FoldTimeTaps(string name, WeightInfo stored)
    {
        long outputs = stored.Shape[0].Fixed;
        long channels = stored.Shape[1].Fixed;
        long taps = stored.Shape[2].Fixed;
        long kh = stored.Shape[3].Fixed;
        long kw = stored.Shape[4].Fixed;
        var shape = new Shape(outputs, taps * channels, kh, kw);
        if (taps == 1)
        {
            return stored.As(name, shape);
        }

        var type = stored.Type.IsQuantized ? DataType.F32 : stored.Type;
        return WeightInfo.Derived(name, type, shape, () =>
        {
            byte[] source = stored.Type.IsQuantized ? F32Bytes(stored) : stored.ReadAll();
            int block = (int)type.ByteSize(kh * kw);
            var folded = new byte[source.Length];
            for (long o = 0; o < outputs; o++)
            {
                for (long c = 0; c < channels; c++)
                {
                    for (long t = 0; t < taps; t++)
                    {
                        long from = ((o * channels + c) * taps + t) * block;
                        long to = ((o * taps + t) * channels + c) * block;
                        Array.Copy(source, from, folded, to, block);
                    }
                }
            }

            return folded;
        });
    }

    private static byte[] F32Bytes(WeightInfo stored)
    {
        var values = GgmlBlockDecoder.Decode(stored.Type, stored.ReadAll());
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
