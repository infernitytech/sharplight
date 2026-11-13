using Sharplight.Graph;

namespace Sharplight.Llm.Serving;

/// <summary>
/// Single-slot <see cref="IBatchDecoder"/> over an <see cref="InferenceSession"/>: inputs of a step run one after another, and
/// a context can only be reset, not truncated, so prefix reuse is limited to whole-context matches. The fallback of
/// <see cref="LlmModel.CreateBatchDecoder"/> for models whose graph has no multi-sequence form. Not thread-safe.
/// </summary>
/// <param name="session">The session that holds the single context.</param>
/// <param name="ownsSession">Dispose <paramref name="session"/> together with the decoder.</param>
public sealed class SessionBatchDecoder(InferenceSession session, bool ownsSession = true) : IBatchDecoder, IDeviceSamplingDecoder
{
    private readonly float[] _logits = new float[session.VocabSize];

    /// <inheritdoc/>
    public int SlotCount => 1;
    /// <inheritdoc/>
    public int ContextLength => session.ContextLength;
    /// <inheritdoc/>
    public int MaxBatchTokens => (int)session.Compiled.Options.MaxTokens;
    /// <inheritdoc/>
    public int VocabSize => session.VocabSize;
    /// <inheritdoc/>
    public bool BatchesSequences => false;
    /// <inheritdoc/>
    public bool CanTruncate => false;
    /// <inheritdoc/>
    public long DeviceMemoryBytes => session.Compiled.AcceleratorMemoryBytes;

    /// <inheritdoc/>
    public int Length(int slot)
    {
        Check(slot);
        return session.Position;
    }

    /// <inheritdoc/>
    public void Reset(int slot)
    {
        Check(slot);
        session.Reset();
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException"><paramref name="length"/> is neither 0 nor the current length.</exception>
    public void Truncate(int slot, int length)
    {
        Check(slot);
        if (length == session.Position)
        {
            return;
        }

        if (length != 0)
        {
            throw new NotSupportedException("This decoder cannot truncate a context.");
        }

        session.Reset();
    }

    /// <inheritdoc/>
    public int MaxLogitsRows => 1;

    /// <inheritdoc/>
    bool IDeviceSamplingDecoder.SamplesOnDevice => session.SamplesOnDevice;

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">An input asks for more than one logits row.</exception>
    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs, IReadOnlyList<SamplingRow?> sampling, Span<int> tokens, Span<float> logprobs)
    {
        var res = new Memory<float>[inputs.Count];
        for (int i = 0; i < inputs.Count; i++)
        {
            Check(inputs[i].Slot);
            if (inputs[i].OutputRows > 1)
            {
                throw new NotSupportedException("This decoder returns the logits of the last row only.");
            }

            if (inputs[i].WantLogits && i < sampling.Count && sampling[i] is { } row)
            {
                (tokens[i], logprobs[i]) = session.ProcessSampled(inputs[i].Tokens.Span, row);
                continue;
            }

            var lg = session.Process(inputs[i].Tokens.Span);
            if (inputs[i].WantLogits)
            {
                lg.CopyTo(_logits, 0);
                res[i] = _logits;
            }
        }
        return res;
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">An input asks for more than one logits row.</exception>
    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs) => Step(inputs, [], [], []);

    private static void Check(int slot)
    {
        if (slot != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(slot));
        }
    }

    /// <summary>Disposes the session when owned.</summary>
    public void Dispose()
    {
        if (ownsSession)
        {
            session.Dispose();
        }
    }
}
