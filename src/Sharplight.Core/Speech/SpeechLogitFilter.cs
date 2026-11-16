namespace Sharplight.Speech;

/// <summary>
/// The logit rules of prompted speech decoding, applied before every token choice: never-generated tokens, tokens not allowed first,
/// and with timestamps the timestamp grammar (timestamps come in pairs around text, never decrease, the window starts with one no later
/// than the initial limit, and a timestamp wins when all timestamps together are likelier than any single text token). These are the rules
/// of the reference Whisper decoders, written against the neutral <see cref="SpeechTokenProtocol"/>. Timestamp rules need a protocol with
/// timestamp tokens.
/// </summary>
internal sealed class SpeechLogitFilter(SpeechTokenProtocol protocol, bool timestamps)
{
    /// <summary>Applies the rules to <paramref name="logits"/> in place.</summary>
    /// <param name="logits">The logits of the next token.</param>
    /// <param name="generated">The tokens generated in this window so far (after the prompt).</param>
    public void Apply(Span<float> logits, IReadOnlyList<int> generated)
    {
        if (generated.Count == 0)
        {
            Suppress(logits, protocol.BeginSuppressIds);
        }

        Suppress(logits, protocol.SuppressIds);
        if (!timestamps)
        {
            return;
        }

        if (protocol.NoTimestampsId is int noTimestamps && noTimestamps < logits.Length)
        {
            logits[noTimestamps] = float.NegativeInfinity;
        }

        int begin = protocol.TimestampBeginId ?? throw new InvalidOperationException("The speech decoder has no timestamp tokens.");
        bool lastWasTimestamp = generated.Count >= 1 && generated[^1] >= begin;
        bool penultimateWasTimestamp = generated.Count < 2 || generated[^2] >= begin;
        if (lastWasTimestamp)
        {
            if (penultimateWasTimestamp)
            {
                // A pair just closed: text (or the end) must follow.
                logits[begin..].Fill(float.NegativeInfinity);
            }
            else
            {
                // A segment's text just ended: only its closing timestamp or the end may follow.
                logits[..protocol.EndId].Fill(float.NegativeInfinity);
            }
        }

        int lastTimestamp = LastTimestamp(generated, begin);
        if (lastTimestamp >= 0)
        {
            // Timestamps never decrease, and a segment has a nonzero length.
            int lowest = lastWasTimestamp && !penultimateWasTimestamp ? lastTimestamp : lastTimestamp + 1;
            logits[begin..Math.Min(lowest, logits.Length)].Fill(float.NegativeInfinity);
        }

        if (generated.Count == 0)
        {
            logits[..begin].Fill(float.NegativeInfinity);
            if (protocol.MaxInitialTimestamp is int maxInitial && begin + maxInitial + 1 < logits.Length)
            {
                logits[(begin + maxInitial + 1)..].Fill(float.NegativeInfinity);
            }
        }

        if (TimestampsLikelier(logits, begin))
        {
            logits[..begin].Fill(float.NegativeInfinity);
        }
    }

    private static void Suppress(Span<float> logits, int[] tokens)
    {
        foreach (int token in tokens)
        {
            if ((uint)token < (uint)logits.Length)
            {
                logits[token] = float.NegativeInfinity;
            }
        }
    }

    private static int LastTimestamp(IReadOnlyList<int> generated, int begin)
    {
        for (int i = generated.Count - 1; i >= 0; i--)
        {
            if (generated[i] >= begin)
            {
                return generated[i];
            }
        }

        return -1;
    }

    /// <summary>True when log Σ p(timestamp) exceeds the largest log p(text token).</summary>
    private static bool TimestampsLikelier(ReadOnlySpan<float> logits, int begin)
    {
        float max = float.NegativeInfinity;
        foreach (float value in logits)
        {
            max = MathF.Max(max, value);
        }

        if (float.IsNegativeInfinity(max))
        {
            return false;
        }

        double timestampSum = 0;
        for (int i = begin; i < logits.Length; i++)
        {
            timestampSum += Math.Exp(logits[i] - max);
        }

        float maxText = float.NegativeInfinity;
        for (int i = 0; i < begin; i++)
        {
            maxText = MathF.Max(maxText, logits[i]);
        }

        // Both sides share the softmax normalizer, so compare log Σ exp(timestamps) with the best text logit.
        return timestampSum > 0 && Math.Log(timestampSum) + max > maxText;
    }
}
