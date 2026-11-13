namespace Sharplight.Llm.Serving;

/// <summary>Single-slot <see cref="IBatchDecoder"/> over an <see cref="InferenceSession"/>.</summary>
public sealed class SessionBatchDecoder(InferenceSession session, bool ownsSession = true) : IBatchDecoder
{
    private readonly float[] _logits = new float[session.VocabSize];

    public int SlotCount => 1;
    public int ContextLength => session.ContextLength;
    public int MaxBatchTokens => (int)session.Compiled.Options.MaxTokens;
    public int VocabSize => session.VocabSize;
    public bool BatchesSequences => false;
    public bool CanTruncate => false;
    public long DeviceMemoryBytes => session.Compiled.AcceleratorMemoryBytes;
    public int Length(int slot)
    {
        Check(slot);
        return session.Position;
    }
    public void Reset(int slot)
    {
        Check(slot);
        session.Reset();
    }
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

    public IReadOnlyList<float[]?> Step(IReadOnlyList<SequenceInput> inputs)
    {
        var res = new float[]?[inputs.Count];
        for (int i = 0; i < inputs.Count; i++)
        {
            Check(inputs[i].Slot);
            var lg = session.Process(inputs[i].Tokens.Span);
            if (inputs[i].WantLogits)
            {
                lg.CopyTo(_logits, 0);
                res[i] = _logits;
            }
        }
        return res;
    }

    private static void Check(int slot)
    {
        if (slot != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(slot));
        }
    }

    public void Dispose()
    {
        if (ownsSession)
        {
            session.Dispose();
        }
    }
}
