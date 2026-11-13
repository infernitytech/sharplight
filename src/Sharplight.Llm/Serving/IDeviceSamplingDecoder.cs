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
    /// Like <see cref="IBatchDecoder.Step"/>; inputs with one logits row (<see cref="SequenceInput.OutputRows"/> = 1) and a
    /// <paramref name="sampling"/> entry get their token chosen on the device (written to <paramref name="tokens"/> /
    /// <paramref name="logprobs"/> at the input's index; the returned logits entry is empty); the other inputs get their logits as
    /// from <see cref="IBatchDecoder.Step"/>.
    /// </summary>
    /// <exception cref="ArgumentException">An input with a sampling entry returns more than one logits row.</exception>
    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs, IReadOnlyList<SamplingRow?> sampling, Span<int> tokens, Span<float> logprobs);
}
