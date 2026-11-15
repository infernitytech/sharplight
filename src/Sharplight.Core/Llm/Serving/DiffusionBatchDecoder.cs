using Sharplight.Llm.Embeddings;

namespace Sharplight.Llm.Serving;

/// <summary>
/// The <see cref="IBatchDecoder"/> of a masked-diffusion model (<see cref="LlmModel.IsDiffusion"/>), from <see cref="LlmModel.CreateBatchDecoder"/>.
/// It holds no per-slot context: the <see cref="InferenceEngine"/> recognizes it and, instead of prefilling and sampling token by token,
/// keeps a <see cref="DiffusionSequence"/> per running request and advances all of them one denoising step per scheduler step with
/// <see cref="Denoise"/> (the model's bidirectional <see cref="Models.WellKnown.DenoiseGraph"/> over each whole sequence, packed into passes of
/// at most <see cref="MaxBatchTokens"/> rows). <see cref="Step"/> is not used. Not thread-safe.
/// </summary>
public sealed class DiffusionBatchDecoder : IBatchDecoder
{
    private readonly PackedBatchRunner _runner;

    internal DiffusionBatchDecoder(PackedBatchRunner runner, int slots, int contextLength, int maskToken, bool shiftLogits)
    {
        _runner = runner;
        SlotCount = slots;
        ContextLength = contextLength;
        MaskToken = maskToken;
        ShiftLogits = shiftLogits;
    }

    /// <inheritdoc/>
    public int SlotCount { get; }
    /// <summary>Longest sequence (prompt plus generated span) of one request.</summary>
    public int ContextLength { get; }
    /// <summary>Most rows (summed over the sequences) of one denoising pass.</summary>
    public int MaxBatchTokens => _runner.MaxBatchTokens;
    /// <inheritdoc/>
    public int VocabSize => _runner.OutputWidth;
    /// <inheritdoc/>
    public bool BatchesSequences => SlotCount > 1;
    /// <inheritdoc/>
    public bool CanTruncate => false;
    /// <inheritdoc/>
    public bool ReusesPrefixes => false;
    /// <inheritdoc/>
    public long DeviceMemoryBytes => _runner.Compiled.AcceleratorMemoryBytes;
    /// <summary>The mask token a generated span starts from.</summary>
    public int MaskToken { get; }
    /// <summary>Whether logits row i predicts position i + 1 (Dream) instead of position i (LLaDA).</summary>
    public bool ShiftLogits { get; }

    /// <inheritdoc/>
    public int Length(int slot) => 0;

    /// <inheritdoc/>
    public void Reset(int slot)
    {
    }

    /// <inheritdoc/>
    public void Truncate(int slot, int length)
    {
    }

    /// <summary>Not used: a diffusion model is driven with <see cref="Denoise"/>.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs)
        => throw new NotSupportedException("A masked-diffusion model has no token-by-token step; the engine drives it with Denoise.");

    /// <summary>
    /// Runs one denoising pass over each sequence (prompt plus span, each at most <see cref="ContextLength"/> tokens) and returns, in input
    /// order, the logits of all rows of each (rows × <see cref="VocabSize"/> floats).
    /// </summary>
    internal float[][] Denoise(IReadOnlyList<DiffusionSequence> sequences)
    {
        var batch = new PackedSequence[sequences.Count];
        for (int i = 0; i < sequences.Count; i++)
        {
            batch[i] = new PackedSequence(sequences[i].Tokens);
        }

        return _runner.Run(batch);
    }

    /// <summary>Releases the compiled graph.</summary>
    public void Dispose() => _runner.Dispose();
}
