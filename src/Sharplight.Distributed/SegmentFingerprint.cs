using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using Sharplight.Models;

namespace Sharplight.Distributed;

/// <summary>
/// A hash of the weights a segment reads, computed the same way on the coordinator and on the worker, so that a worker refuses to run
/// layers of a model copy that differs from the coordinator's (another model, quantization or revision) instead of producing wrong tokens.
/// </summary>
internal static class SegmentFingerprint
{
    private const int SampleBytes = 4096;

    /// <summary>The fingerprint of <paramref name="weights"/> (order does not matter).</summary>
    public static string Compute(IEnumerable<WeightInfo> weights, WeightVerification verification)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> number = stackalloc byte[8];
        var sample = new byte[SampleBytes];
        foreach (var w in weights.DistinctBy(w => w.Name).OrderBy(w => w.Name, StringComparer.Ordinal))
        {
            hash.AppendData(Encoding.UTF8.GetBytes($"{w.Name}|{w.Type}|{w.Shape}|"));
            long length = w.Source.ByteLength;
            BinaryPrimitives.WriteInt64LittleEndian(number, length);
            hash.AppendData(number);
            if (verification == WeightVerification.Full)
            {
                AppendAll(hash, w.Source);
            }
            else if (verification == WeightVerification.Sampled && IsStored(w.Source))
            {
                foreach (long offset in new[] { 0, Math.Max(0, length / 2 - SampleBytes / 2), Math.Max(0, length - SampleBytes) }.Distinct())
                {
                    int count = (int)Math.Min(SampleBytes, length - offset);
                    w.Source.Read(offset, sample.AsSpan(0, count));
                    hash.AppendData(sample, 0, count);
                }
            }
        }

        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }

    /// <summary>
    /// Whether reading a sample is cheap: the weight lies in a file or in memory, or is a view, stack or concatenation of stored weights
    /// (a fused projection); a derived weight (a conversion) would be computed whole, so only its metadata counts.
    /// </summary>
    private static bool IsStored(IWeightSource source)
        => source is SliceWeightSource or StackedWeightSource or ConcatenatedWeightSource
           || source.TryGetFileRange(out _, out _) || source.TryGetMapping(out _, out _) || source.TryGetPointer() != 0;

    private static void AppendAll(IncrementalHash hash, IWeightSource source)
    {
        var chunk = new byte[8 << 20];
        for (long offset = 0; offset < source.ByteLength; offset += chunk.Length)
        {
            int count = (int)Math.Min(chunk.Length, source.ByteLength - offset);
            source.Read(offset, chunk.AsSpan(0, count));
            hash.AppendData(chunk, 0, count);
        }
    }
}
