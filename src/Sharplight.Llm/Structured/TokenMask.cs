using System.Numerics;

namespace Sharplight.Llm.Structured;

/// <summary>
/// A set of allowed token ids over a vocabulary, stored as a packed bitset (bit <c>i % 32</c> of word <c>i / 32</c> set =
/// token <c>i</c> allowed). The packed layout is what a device-side sampler takes as its mask input; on the CPU
/// <see cref="Apply"/> writes negative infinity into the logits of disallowed tokens. Not thread-safe.
/// </summary>
public sealed class TokenMask
{
    private readonly uint[] _words;

    /// <summary>Creates an empty mask (nothing allowed).</summary>
    /// <param name="vocabSize">Number of token ids covered.</param>
    public TokenMask(int vocabSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(vocabSize);
        VocabSize = vocabSize;
        _words = new uint[(vocabSize + 31) / 32];
    }

    /// <summary>Number of token ids covered.</summary>
    public int VocabSize { get; }

    /// <summary>The packed bits, <c>ceil(VocabSize / 32)</c> words; bits past <see cref="VocabSize"/> are always zero.</summary>
    public ReadOnlySpan<uint> Words => _words;

    /// <summary>Writable bits for matchers that copy whole cached masks; callers keep bits past <see cref="VocabSize"/> zero.</summary>
    internal Span<uint> WritableWords => _words;

    /// <summary>Disallows every token.</summary>
    public void Clear() => Array.Clear(_words);

    /// <summary>Allows every token.</summary>
    public void AllowAll()
    {
        Array.Fill(_words, uint.MaxValue);
        int tail = VocabSize % 32;
        if (tail != 0)
        {
            _words[^1] = (1u << tail) - 1;
        }
    }

    /// <summary>Allows <paramref name="token"/> (ids outside the vocabulary are ignored).</summary>
    public void Allow(int token)
    {
        if ((uint)token < (uint)VocabSize)
        {
            _words[token >> 5] |= 1u << (token & 31);
        }
    }

    /// <summary>Disallows <paramref name="token"/> (ids outside the vocabulary are ignored).</summary>
    public void Disallow(int token)
    {
        if ((uint)token < (uint)VocabSize)
        {
            _words[token >> 5] &= ~(1u << (token & 31));
        }
    }

    /// <summary>Whether <paramref name="token"/> is allowed.</summary>
    public bool IsAllowed(int token) => (uint)token < (uint)VocabSize && (_words[token >> 5] & (1u << (token & 31))) != 0;

    /// <summary>Adds every token allowed by <paramref name="other"/> (same vocabulary size).</summary>
    /// <exception cref="ArgumentException">The vocabulary sizes differ.</exception>
    public void UnionWith(TokenMask other)
    {
        if (other.VocabSize != VocabSize)
        {
            throw new ArgumentException("Masks cover different vocabularies.", nameof(other));
        }

        for (int i = 0; i < _words.Length; i++)
        {
            _words[i] |= other._words[i];
        }
    }

    /// <summary>Number of allowed tokens.</summary>
    public int Count()
    {
        int n = 0;
        foreach (uint w in _words)
        {
            n += BitOperations.PopCount(w);
        }

        return n;
    }

    /// <summary>Sets the logits of all disallowed tokens to negative infinity.</summary>
    /// <param name="logits">One row of logits; entries past <see cref="VocabSize"/> are disallowed as well.</param>
    public void Apply(Span<float> logits)
    {
        int n = Math.Min(logits.Length, VocabSize);
        for (int w = 0; w * 32 < n; w++)
        {
            uint bits = _words[w];
            if (bits == uint.MaxValue)
            {
                continue;
            }

            int start = w * 32;
            int end = Math.Min(start + 32, n);
            if (bits == 0)
            {
                logits[start..end].Fill(float.NegativeInfinity);
                continue;
            }

            for (int i = start; i < end; i++)
            {
                if ((bits & (1u << (i - start))) == 0)
                {
                    logits[i] = float.NegativeInfinity;
                }
            }
        }

        if (logits.Length > n)
        {
            logits[n..].Fill(float.NegativeInfinity);
        }
    }
}
