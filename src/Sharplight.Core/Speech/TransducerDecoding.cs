namespace Sharplight.Speech;

/// <summary>
/// Greedy transducer decoding on the host (RNN-T and the token-and-duration transducer, TDT), as NeMo and transformers decode: at the current
/// encoder row the joint network scores the tokens (and, for TDT, the durations); a token other than blank is emitted and advances the
/// prediction network; the row advances by one on blank (RNN-T) or by the predicted duration (TDT, at least one row on blank). A row emits at
/// most <see cref="FrameDecodingSettings.MaxSymbolsPerStep"/> tokens before the decoding moves on.
/// </summary>
internal static class TransducerDecoding
{
    /// <summary>Decodes the projected encoder rows of one window.</summary>
    /// <param name="network">The prediction and joint networks.</param>
    /// <param name="settings">The decoding settings.</param>
    /// <param name="rows">The projected encoder rows, row-major [count, <see cref="TransducerNetwork.Width"/>].</param>
    /// <param name="cancellationToken">Stops between steps.</param>
    public static List<AlignedToken> Greedy(TransducerNetwork network, FrameDecodingSettings settings, ReadOnlySpan<float> rows, CancellationToken cancellationToken)
    {
        int width = network.Width;
        int count = rows.Length / width;
        int durations = settings.Decoding == "tdt" ? settings.Durations.Length : 0;
        int vocabulary = network.Outputs - durations;
        var tokens = new List<AlignedToken>();
        network.Reset(settings.Blank);
        int row = 0;
        int symbols = 0;
        while (row < count)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var logits = network.Joint(rows.Slice(row * width, width));
            var tokenLogits = logits.AsSpan(0, vocabulary);
            int token = ArgMax(tokenLogits);
            int advance;
            if (durations > 0)
            {
                advance = settings.Durations[ArgMax(logits.AsSpan(vocabulary, durations))];
                if (token == settings.Blank && advance == 0)
                {
                    advance = 1;
                }
            }
            else
            {
                advance = token == settings.Blank ? 1 : 0;
            }

            if (token != settings.Blank)
            {
                tokens.Add(new AlignedToken(token, row, Math.Max(advance, 1), (float)LogProbabilities.Of(tokenLogits, token)));
                network.Predict(token);
            }

            // Too many tokens at one row: move on (NeMo's max_symbols_per_step).
            symbols = advance == 0 ? symbols + 1 : 0;
            if (symbols >= settings.MaxSymbolsPerStep)
            {
                advance = 1;
                symbols = 0;
            }

            row += advance;
        }

        return tokens;
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }
}
