using System.Text;
using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Deterministic multi-slot decoder: the next token of a slot is a pure function of its context (<see cref="Next"/>), the logits
/// are a one-hot spike on it. Records every step so tests can assert on batching.
/// </summary>
public sealed class FakeBatchDecoder : IBatchDecoder
{
    public const int Marker = 1;   // prompts end with this token; scripted replies are indexed relative to it
    private readonly List<int>[] _ctx;
    private readonly float[][] _logits;

    public FakeBatchDecoder(int slots, int contextLength = 4096, int maxBatchTokens = 64, bool batches = true, bool canTruncate = true)
    {
        SlotCount = slots;
        ContextLength = contextLength;
        MaxBatchTokens = maxBatchTokens;
        BatchesSequences = batches;
        CanTruncate = canTruncate;
        _ctx = [.. Enumerable.Range(0, slots).Select(_ => new List<int>())];
        _logits = [.. Enumerable.Range(0, 64).Select(_ => new float[VocabSize])];
        Next = ctx => (ctx[^1] + 1) % 256;
    }

    public int SlotCount { get; }
    public int ContextLength { get; }
    public int MaxBatchTokens { get; }
    public int VocabSize => 260;
    public bool BatchesSequences { get; }
    public bool CanTruncate { get; }
    public Func<IReadOnlyList<int>, int> Next { get; set; }
    public TimeSpan StepDelay { get; set; }
    public int FailAfterSteps { get; set; } = -1;
    public List<(int Slot, int Rows, bool Logits)[]> Steps { get; } = [];
    public List<(int Slot, int Length)> Truncations { get; } = [];
    public List<int> Resets { get; } = [];

    /// <summary>Replies with <paramref name="reply"/> (then end-of-generation) after a prompt ending in <see cref="Marker"/>; repeats if <paramref name="loop"/>.</summary>
    public void Script(string reply, bool loop = false)
    {
        var bytes = Encoding.UTF8.GetBytes(reply);
        Next = ctx =>
        {
            int m = -1;
            for (int i = ctx.Count - 1; i >= 0; i--)
            {
                if (ctx[i] == Marker)
                {
                    m = i;
                    break;
                }
            }

            int k = m < 0 ? 0 : ctx.Count - m - 1;
            if (loop && bytes.Length > 0)
            {
                k %= bytes.Length;
            }

            return k < bytes.Length ? bytes[k] : FakeTokenizer.Eos;
        };
    }

    public int Length(int slot) => _ctx[slot].Count;
    public void Reset(int slot) { _ctx[slot].Clear(); lock (Resets)
        {
            Resets.Add(slot);
        }
    }
    public void Truncate(int slot, int length)
    {
        if (!CanTruncate)
        {
            throw new NotSupportedException();
        }

        _ctx[slot].RemoveRange(length, _ctx[slot].Count - length);
        lock (Truncations)
        {
            Truncations.Add((slot, length));
        }
    }
    public IReadOnlyList<int> Context(int slot) => _ctx[slot];

    public IReadOnlyList<float[]?> Step(IReadOnlyList<SequenceInput> inputs)
    {
        if (inputs.Select(i => i.Slot).Distinct().Count() != inputs.Count)
        {
            throw new InvalidOperationException("two inputs for one slot");
        }

        if (inputs.Sum(i => i.Tokens.Length) > MaxBatchTokens)
        {
            throw new InvalidOperationException("batch too large");
        }

        if (!BatchesSequences && inputs.Count > 1)
        {
            /* executed one after another;
            still recorded as one step */
        }
        lock (Steps)
        {
            if (FailAfterSteps >= 0 && Steps.Count >= FailAfterSteps)
            {
                throw new InvalidOperationException("injected failure");
            }

            Steps.Add([.. inputs.Select(i => (i.Slot, i.Tokens.Length, i.WantLogits))]);
        }
        if (StepDelay > TimeSpan.Zero)
        {
            Thread.Sleep(StepDelay);
        }

        var res = new float[inputs.Count][];
        for (int i = 0; i < inputs.Count; i++)
        {
            var c = _ctx[inputs[i].Slot];
            if (c.Count + inputs[i].Tokens.Length > ContextLength)
            {
                throw new InvalidOperationException("context overflow");
            }

            c.AddRange(inputs[i].Tokens.ToArray());
            if (!inputs[i].WantLogits)
            {
                continue;
            }

            var l = _logits[i];
            Array.Fill(l, -10f);
            l[Next(c)] = 10f;
            res[i] = l;
        }
        return res;
    }

    public void Dispose() { }
}
