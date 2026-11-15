using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// ONNX StringSplit (see <c>StandardOps.StringSplit</c>): splits every string like Python's <c>str.split(delimiter, maxsplit)</c>, writes
/// the substrings token-major padded with empty strings, their counts, and publishes the largest count.
/// </summary>
internal sealed unsafe class StringSplitOp(TRef x, TRef tokens, TRef counts, string delimiter, long maxSplit, string countSymbol, long bound)
    : CpuOp
{
    public override string Name => "StringSplit";

    public override void Run(RunParameters p)
    {
        long n = x.Elements(p);
        var parts = new List<string>[n];
        int largest = 0;
        for (long i = 0; i < n; i++)
        {
            parts[i] = Split(p.Strings[((int*)x.P)[i]], delimiter, maxSplit);
            largest = Math.Max(largest, parts[i].Count);
            ((int*)counts.P)[i] = parts[i].Count;
        }

        if (largest > bound)
        {
            throw new InvalidOperationException(
                $"StringSplit produced {largest} substrings of one string, more than the compiled bound {bound} (OnnxImportOptions.StringSplitBound).");
        }

        int empty = p.Strings.Add("");
        for (int k = 0; k < largest; k++)
        {
            for (long i = 0; i < n; i++)
            {
                ((int*)tokens.P)[k * n + i] = k < parts[i].Count ? p.Strings.Add(parts[i][k]) : empty;
            }
        }

        p.Symbols[countSymbol] = largest;
    }

    /// <summary>Python's <c>str.split</c>: on the delimiter, or on runs of whitespace when it is empty; at most <paramref name="maxSplit"/> splits when non-negative.</summary>
    internal static List<string> Split(string s, string delimiter, long maxSplit)
    {
        int limit = maxSplit < 0 ? int.MaxValue : (int)Math.Min(maxSplit, int.MaxValue - 1) + 1;
        if (delimiter.Length > 0)
        {
            return [.. s.Split(delimiter, limit)];
        }

        var result = new List<string>();
        int i = 0;
        while (true)
        {
            while (i < s.Length && char.IsWhiteSpace(s[i]))
            {
                i++;
            }

            if (i == s.Length)
            {
                return result;
            }

            if (result.Count == limit - 1)
            {
                // The last split keeps the remainder (with its trailing whitespace), like Python.
                result.Add(s[i..]);
                return result;
            }

            int start = i;
            while (i < s.Length && !char.IsWhiteSpace(s[i]))
            {
                i++;
            }

            result.Add(s[start..i]);
        }
    }
}
