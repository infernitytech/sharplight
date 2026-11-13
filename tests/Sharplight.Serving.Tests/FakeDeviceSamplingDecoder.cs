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
    public IReadOnlyList<float[]?> Step(IReadOnlyList<SequenceInput> inputs) => inner.Step(inputs);

    public IReadOnlyList<float[]?> Step(IReadOnlyList<SequenceInput> inputs, IReadOnlyList<SamplingRow?> sampling, Span<int> tokens, Span<float> logprobs)
    {
        var logits = inner.Step(inputs).ToArray();
        for (int i = 0; i < inputs.Count; i++)
        {
            if (!inputs[i].WantLogits || i >= sampling.Count || sampling[i] is not { } row)
            {
                continue;
            }

            SampledRows.Add((inputs[i].Slot, row));
            tokens[i] = Array.IndexOf(logits[i]!, logits[i]!.Max());
            logprobs[i] = -0.5f;
            logits[i] = null;
        }

        return logits;
    }

    public void Dispose() => inner.Dispose();
}
