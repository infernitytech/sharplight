using Sharplight.Graph;
using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// <see cref="FakeBatchDecoder"/> that also "samples on the device": rows with a sampling entry get the arg-max of the fake logits
/// (log-probability -0.5) and no logits. Records every sampling row it was given, in order.
/// </summary>
internal sealed class FakeDeviceSamplingDecoder(FakeBatchDecoder inner) : IBatchDecoder, IDeviceSamplingDecoder
{
    public FakeBatchDecoder Inner => inner;
    public List<(int Slot, SamplingRow Row)> SampledRows { get; } = [];
    public int SlotCount => inner.SlotCount;
    public int ContextLength => inner.ContextLength;
    public int MaxBatchTokens => inner.MaxBatchTokens;
    public int VocabSize => inner.VocabSize;
    public bool BatchesSequences => inner.BatchesSequences;
    public bool CanTruncate => inner.CanTruncate;
    public bool SamplesOnDevice => true;
    public int Length(int slot) => inner.Length(slot);
    public void Reset(int slot) => inner.Reset(slot);
    public void Truncate(int slot, int length) => inner.Truncate(slot, length);
    public int MaxLogitsRows => inner.MaxLogitsRows;
    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs) => inner.Step(inputs);

    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs, IReadOnlyList<SamplingRow?> sampling, Span<int> tokens, Span<float> logprobs)
    {
        var logits = inner.Step(inputs).ToArray();
        for (int i = 0; i < inputs.Count; i++)
        {
            if (!inputs[i].WantLogits || i >= sampling.Count || sampling[i] is not { } row)
            {
                continue;
            }

            if (inputs[i].OutputRows != 1)
            {
                throw new ArgumentException("Device sampling needs exactly one logits row.");
            }

            SampledRows.Add((inputs[i].Slot, row));
            var span = logits[i].Span;
            int best = 0;
            for (int t = 1; t < span.Length; t++)
            {
                best = span[t] > span[best] ? t : best;
            }

            tokens[i] = best;
            logprobs[i] = -0.5f;
            logits[i] = Memory<float>.Empty;
        }

        return logits;
    }

    public void Dispose() => inner.Dispose();
}
