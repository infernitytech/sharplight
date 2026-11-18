using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;

namespace Sharplight.Llm.Serving;

/// <summary>
/// Content hashes of KV blocks for prefix caching: the hash of block j is SHA-256 (truncated to 128 bits) of the hash of block j - 1,
/// the block's tokens and the hashes of the non-text content overlapping it; the chain starts at a seed made of the request's cache
/// namespace (partition and salt). Equal hashes therefore mean equal prefixes in the same namespace, and a collision-resistant hash
/// keeps one prompt from being crafted to alias another's blocks.
/// </summary>
internal static class KvBlockHash
{
    private const string Version = "sharplight-kv-prefix-v1";

    /// <summary>The chain's start for a cache namespace (null and empty partitions or salts are different namespaces).</summary>
    public static UInt128 Seed(string? partition, string? salt)
    {
        var sb = new StringBuilder(Version);
        Append(sb, partition);
        Append(sb, salt);
        return Truncate(SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString())));

        static void Append(StringBuilder sb, string? s)
        {
            sb.Append('\0');
            if (s is null)
            {
                sb.Append('n');
            }
            else
            {
                sb.Append('s').Append(s.Length).Append(':').Append(s);
            }
        }
    }

    /// <summary>The hash of the block holding <paramref name="tokens"/> at positions <paramref name="start"/> .. after the block hashed <paramref name="parent"/>.</summary>
    public static UInt128 Next(UInt128 parent, ReadOnlySpan<int> tokens, int start, IReadOnlyList<PromptContentKey> content)
    {
        int extras = 0;
        foreach (var c in content)
        {
            extras += Overlaps(c, start, tokens.Length) ? 1 : 0;
        }

        int bytes = 16 + tokens.Length * 4 + extras * 24;
        Span<byte> buf = bytes <= 1024 ? stackalloc byte[bytes] : new byte[bytes];
        BinaryPrimitives.WriteUInt128LittleEndian(buf, parent);
        for (int i = 0; i < tokens.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(buf[(16 + i * 4)..], tokens[i]);
        }

        int o = 16 + tokens.Length * 4;
        foreach (var c in content)
        {
            if (Overlaps(c, start, tokens.Length))
            {
                BinaryPrimitives.WriteInt32LittleEndian(buf[o..], c.Start);
                BinaryPrimitives.WriteInt32LittleEndian(buf[(o + 4)..], c.Length);
                BinaryPrimitives.WriteUInt128LittleEndian(buf[(o + 8)..], c.Hash);
                o += 24;
            }
        }

        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(buf, hash);
        return Truncate(hash);
    }

    /// <summary>Appends to <paramref name="hashes"/> the hashes of the full blocks of <paramref name="tokens"/> after the ones it holds, up to <paramref name="limit"/> tokens.</summary>
    public static void Extend(List<UInt128> hashes, UInt128 seed, IReadOnlyList<int> tokens, int limit, int blockSize, IReadOnlyList<PromptContentKey> content)
    {
        int full = Math.Min(limit, tokens.Count) / blockSize;
        if (hashes.Count >= full)
        {
            return;
        }

        Span<int> block = blockSize <= 256 ? stackalloc int[blockSize] : new int[blockSize];
        for (int j = hashes.Count; j < full; j++)
        {
            for (int i = 0; i < blockSize; i++)
            {
                block[i] = tokens[j * blockSize + i];
            }

            hashes.Add(Next(j == 0 ? seed : hashes[j - 1], block, j * blockSize, content));
        }
    }

    private static bool Overlaps(PromptContentKey c, int start, int length) => c.Start < start + length && c.Start + c.Length > start;

    private static UInt128 Truncate(ReadOnlySpan<byte> sha) => BinaryPrimitives.ReadUInt128LittleEndian(sha);
}
