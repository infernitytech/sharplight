using System.Runtime.InteropServices;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Encodes f32 values as ggml Q8_0 blocks (32 values: an f16 scale max|x| / 127, then 32 int8 values round(x / scale)), as llama.cpp's
/// reference quantizer does. Used to keep a LoRA-merged quantized weight quantized (8.5 bits per value instead of f16's 16).
/// </summary>
internal static class Q8Blocks
{
    /// <summary>Values per block.</summary>
    public const int BlockSize = 32;

    private const int BlockBytes = 34;

    /// <summary>The Q8_0 bytes of <paramref name="values"/> (a multiple of <see cref="BlockSize"/> values).</summary>
    /// <param name="values">The values, row-major; every row a multiple of 32 long.</param>
    /// <exception cref="ArgumentException">The count is not a multiple of 32.</exception>
    public static byte[] Encode(ReadOnlySpan<float> values)
    {
        if (values.Length % BlockSize != 0)
        {
            throw new ArgumentException($"Q8_0 needs a multiple of {BlockSize} values, got {values.Length}.", nameof(values));
        }

        int blocks = values.Length / BlockSize;
        var bytes = new byte[blocks * BlockBytes];
        var source = values.ToArray();
        Parallel.For(0, blocks, b =>
        {
            var block = source.AsSpan(b * BlockSize, BlockSize);
            float max = 0f;
            foreach (float v in block)
            {
                max = Math.Max(max, Math.Abs(v));
            }

            float scale = max / 127f;
            float inverse = scale == 0f ? 0f : 1f / scale;
            var destination = bytes.AsSpan(b * BlockBytes, BlockBytes);
            MemoryMarshal.Write(destination, (Half)scale);
            for (int i = 0; i < BlockSize; i++)
            {
                destination[2 + i] = unchecked((byte)(sbyte)Math.Round(block[i] * inverse, MidpointRounding.AwayFromZero));
            }
        });
        return bytes;
    }
}
