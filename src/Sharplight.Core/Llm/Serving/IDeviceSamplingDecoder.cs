using Sharplight.Graph;

namespace Sharplight.Llm.Serving;

/// <summary>
/// An <see cref="IBatchDecoder"/> whose compiled graph can choose tokens on the device (a <see cref="SamplingHead"/>), so a step
/// downloads token ids instead of logits for the rows sampled there.
/// </summary>
internal interface IDeviceSamplingDecoder
{
    /// <summary>True when <see cref="Step"/> can sample rows on the device.</summary>
    public bool SamplesOnDevice { get; }

    /// <summary>
    /// Like <see cref="IBatchDecoder.Step"/>; inputs with a <paramref name="sampling"/> entry get a token chosen on the device for
    /// each of their logits rows (row j of an input with the entry's <see cref="SamplingRow.Counter"/> + j, so speculative
    /// verification rows draw exactly what decoding them one by one would); the tokens and log-probabilities are written to
    /// <paramref name="tokens"/> / <paramref name="logprobs"/> by logits row (the inputs' rows consecutively, in input order) and
    /// the input's logits entry is empty. The other inputs get their logits as from <see cref="IBatchDecoder.Step"/>.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="tokens"/> or <paramref name="logprobs"/> is shorter than the step's logits rows.</exception>
    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs, IReadOnlyList<SamplingRow?> sampling, Span<int> tokens, Span<float> logprobs);
}
