using System.Runtime.InteropServices;
using Sharplight.Distributed.Protocol;

namespace Sharplight.Distributed.Tests;

/// <summary>The 16-bit wire forms of boundary values: rounding, special values and lengths.</summary>
public class BoundaryCodecTests
{
    [Theory]
    [InlineData(BoundaryPrecision.Float16)]
    [InlineData(BoundaryPrecision.BFloat16)]
    [InlineData(BoundaryPrecision.Float32)]
    public void RoundTripsWithinThePrecision(BoundaryPrecision precision)
    {
        var rng = new Random(3);
        var values = Enumerable.Range(0, 1037).Select(_ => (float)(rng.NextDouble() * 20 - 10)).ToArray();
        var source = MemoryMarshal.AsBytes(values.AsSpan());
        var wire = new byte[BoundaryCodec.EncodedLength(source.Length, precision)];
        BoundaryCodec.Encode(source, wire, precision);
        var back = new float[values.Length];
        BoundaryCodec.Decode(wire, MemoryMarshal.AsBytes(back.AsSpan()), precision);
        float tolerance = precision switch
        {
            BoundaryPrecision.Float16 => 1f / 2048,
            BoundaryPrecision.BFloat16 => 1f / 256,
            _ => 0,
        };
        for (int i = 0; i < values.Length; i++)
        {
            Assert.True(Math.Abs(back[i] - values[i]) <= tolerance * Math.Abs(values[i]), $"{values[i]} -> {back[i]}");
        }
    }

    [Fact]
    public void Float16MatchesTheHalfConversion()
    {
        float[] values = [0f, -0f, 1f, 65504f, 1e6f, -1e6f, 6e-8f, float.PositiveInfinity, float.NaN, 0.1f, 1.0009765625f];
        var wire = new byte[values.Length * 2];
        BoundaryCodec.Encode(MemoryMarshal.AsBytes(values.AsSpan()), wire, BoundaryPrecision.Float16);
        var halves = MemoryMarshal.Cast<byte, Half>(wire);
        for (int i = 0; i < values.Length; i++)
        {
            Assert.Equal(BitConverter.HalfToUInt16Bits((Half)values[i]), BitConverter.HalfToUInt16Bits(halves[i]));
        }
    }

    [Theory]
    [InlineData(0x3F80_0000u, (ushort)0x3F80)] // 1
    [InlineData(0x3F80_8000u, (ushort)0x3F80)] // halfway, even below: stays
    [InlineData(0x3F81_8000u, (ushort)0x3F82)] // halfway, odd below: rounds up
    [InlineData(0x3F80_8001u, (ushort)0x3F81)] // above halfway
    [InlineData(0x7F7F_FFFFu, (ushort)0x7F80)] // f32 max rounds to infinity
    [InlineData(0x7F80_0000u, (ushort)0x7F80)] // infinity
    [InlineData(0xFF80_0000u, (ushort)0xFF80)] // -infinity
    [InlineData(0x7F80_0001u, (ushort)0x7FC0)] // a signalling NaN stays a NaN
    [InlineData(0x7FFF_FFFFu, (ushort)0x7FFF)] // a NaN with all bits never wraps to zero
    public void BFloat16RoundsToNearestEven(uint bits, ushort expected) => Assert.Equal(expected, BoundaryCodec.ToBFloat16(bits));

    [Fact]
    public void DecodeFillsAShorterDestination()
    {
        float[] values = [1, 2, 3, 4];
        var wire = new byte[8];
        BoundaryCodec.Encode(MemoryMarshal.AsBytes(values.AsSpan()), wire, BoundaryPrecision.BFloat16);
        var back = new float[2];
        BoundaryCodec.Decode(wire, MemoryMarshal.AsBytes(back.AsSpan()), BoundaryPrecision.BFloat16);
        Assert.Equal([1f, 2f], back);
    }

    [Theory]
    [InlineData(null, BoundaryPrecision.Float32)]
    [InlineData("f32", BoundaryPrecision.Float32)]
    [InlineData("F16", BoundaryPrecision.Float16)]
    [InlineData("bf16", BoundaryPrecision.BFloat16)]
    public void ParsesPrecisionNames(string? name, BoundaryPrecision expected) => Assert.Equal(expected, RemoteSegmentOptions.ParseBoundaryPrecision(name));

    [Fact]
    public void RefusesAnUnknownPrecisionName() => Assert.Throws<ArgumentException>(() => RemoteSegmentOptions.ParseBoundaryPrecision("f8"));
}
