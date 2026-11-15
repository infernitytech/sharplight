using Sharplight.Llm.Embeddings;
using Sharplight.Tokenizers;

namespace Sharplight.Llm;

/// <summary>
/// Masked-diffusion decoding (Dream, LLaDA): the prompt is followed by <see cref="GenerationOptions.MaxNewTokens"/> mask tokens, and each
/// step runs the model's bidirectional <see cref="Models.WellKnown.DenoiseGraph"/> over the whole sequence and reveals the masked positions it
/// is most confident about with their most likely token (greedy), as scheduled by <see cref="DiffusionOptions"/>. Obtain it from
/// <see cref="LlmModel.CreateDiffusionGenerator"/>. Not thread-safe.
/// </summary>
public sealed class DiffusionGenerator : IDisposable
{
    private readonly PackedBatchRunner _runner;
    private readonly ITokenizer _tokenizer;
    private readonly int _maskToken;
    private readonly bool _shiftLogits;
    private readonly int _vocab;

    internal DiffusionGenerator(PackedBatchRunner runner, ITokenizer tokenizer, int maskToken, bool shiftLogits)
    {
        _runner = runner;
        _tokenizer = tokenizer;
        _maskToken = maskToken;
        _shiftLogits = shiftLogits;
        _vocab = runner.OutputWidth;
    }

    /// <summary>Longest sequence (prompt plus generated tokens) one pass handles.</summary>
    public int ContextLength => _runner.MaxBatchTokens;

    /// <summary>The mask token the generated span starts from.</summary>
    public int MaskToken => _maskToken;

    /// <summary>
    /// Generates <see cref="GenerationOptions.MaxNewTokens"/> tokens after <paramref name="prompt"/> and returns them, cut before the first
    /// end-of-generation token when <see cref="GenerationOptions.StopAtEog"/> is set. <paramref name="onStep"/> sees the generated span
    /// (mask tokens where nothing is revealed yet) after every step.
    /// </summary>
    /// <exception cref="ArgumentException">The prompt is empty, the sequence does not fit <see cref="ContextLength"/>, or the block schedule does not divide the span or the steps.</exception>
    public int[] Generate(ReadOnlySpan<int> prompt, GenerationOptions options, Action<ReadOnlyMemory<int>>? onStep = null)
    {
        int span = options.MaxNewTokens;
        if (prompt.Length + span > ContextLength)
        {
            throw new ArgumentException($"Prompt and generated span ({prompt.Length + span} tokens) exceed the context ({ContextLength}).");
        }

        var sequence = new DiffusionSequence(prompt, span, options.Diffusion, _maskToken, _shiftLogits);
        while (!sequence.Done)
        {
            var logits = _runner.Run([new PackedSequence(sequence.Tokens)])[0];
            sequence.Apply(logits, _vocab);
            onStep?.Invoke(sequence.Generated);
        }

        var generated = sequence.Generated.Span;
        if (options.StopAtEog)
        {
            for (int i = 0; i < generated.Length; i++)
            {
                if (_tokenizer.IsEndOfGeneration(generated[i]))
                {
                    return generated[..i].ToArray();
                }
            }
        }

        return generated.ToArray();
    }

    /// <summary>Generates for <paramref name="prompt"/> (encoded with special tokens) and returns the decoded text.</summary>
    public string Generate(string prompt, GenerationOptions options, bool parseSpecial = true)
        => _tokenizer.Decode(Generate(_tokenizer.Encode(prompt, addSpecial: true, parseSpecial: parseSpecial), options));

    /// <summary>Releases the compiled graph.</summary>
    public void Dispose() => _runner.Dispose();
}
