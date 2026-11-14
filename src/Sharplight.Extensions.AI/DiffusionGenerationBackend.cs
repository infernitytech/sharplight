using System.Runtime.CompilerServices;
using Sharplight.Llm;

namespace Sharplight.Extensions.AI;

/// <summary>
/// <see cref="ITextGenerationBackend"/> over a masked-diffusion <see cref="LlmModel"/> (<see cref="LlmModel.IsDiffusion"/>: Dream, LLaDA): each
/// request denoises a span of <see cref="GenerationOptions.MaxNewTokens"/> tokens (capped to the context) with a <see cref="DiffusionGenerator"/>
/// on the schedule of <see cref="GenerationOptions.Diffusion"/>, and the text is streamed as one piece once the span is complete (the tokens
/// are not final before the last step). Generations are serialized; the denoising graph is compiled on the first request.
/// <see cref="SharplightChatClient"/> picks this backend for diffusion models.
/// </summary>
public sealed class DiffusionGenerationBackend : ITextGenerationBackend
{
    private readonly LlmModel _model;
    private readonly bool _ownsModel;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private DiffusionGenerator? _generator;
    private int _disposed;

    /// <summary>Wraps <paramref name="model"/>; <paramref name="ownsModel"/> disposes it with the backend.</summary>
    /// <exception cref="ArgumentException"><paramref name="model"/> is not a masked-diffusion model.</exception>
    public DiffusionGenerationBackend(LlmModel model, bool ownsModel = false)
    {
        if (!model.IsDiffusion)
        {
            throw new ArgumentException($"{model.Definition.Name} is not a masked-diffusion model.", nameof(model));
        }

        _model = model;
        _ownsModel = ownsModel;
        ContextLength = int.MaxValue; // refined once the generator exists
    }

    /// <summary>The wrapped model.</summary>
    public LlmModel Model => _model;
    /// <inheritdoc/>
    public string ModelName => _model.Definition.Name;
    /// <inheritdoc/>
    public string? ChatTemplate => _model.ChatTemplate;
    /// <inheritdoc/>
    public string? BosToken => _model.Tokenizer.BosId is int i ? _model.Tokenizer.TokenString(i) : null;
    /// <inheritdoc/>
    public string? EosToken => _model.Tokenizer.EosId is int i ? _model.Tokenizer.TokenString(i) : null;
    /// <inheritdoc/>
    public bool SupportsImages => false;
    /// <inheritdoc/>
    public int ContextLength { get; private set; }

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> GenerateAsync(TextGenerationRequest request, TextGenerationResult result,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (request.Images.Count > 0)
        {
            throw new NotSupportedException("Masked-diffusion models take text prompts only.");
        }

        string text;
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            text = await Task.Factory.StartNew(() => Run(request, result, cancellationToken), cancellationToken,
                TaskCreationOptions.LongRunning, TaskScheduler.Default).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }

        if (text.Length > 0)
        {
            yield return text;
        }
    }

    private string Run(TextGenerationRequest request, TextGenerationResult result, CancellationToken ct)
    {
        var generator = _generator ??= _model.CreateDiffusionGenerator();
        ContextLength = generator.ContextLength;
        var tokenizer = _model.Tokenizer;
        // The chat template emits BOS itself, so no special tokens are added here.
        var prompt = tokenizer.Encode(request.Prompt, addSpecial: false, parseSpecial: true);
        if (prompt.Length >= generator.ContextLength)
        {
            throw new InvalidOperationException($"The prompt ({prompt.Length} tokens) does not fit the context ({generator.ContextLength}).");
        }

        var options = request.Options with { MaxNewTokens = Math.Min(request.Options.MaxNewTokens, generator.ContextLength - prompt.Length) };
        var ids = generator.Generate(prompt, options, _ => ct.ThrowIfCancellationRequested());
        bool cut = ids.Length < options.MaxNewTokens;
        result.PromptTokens = prompt.Length;
        // A span cut at an end-of-generation token counts that token, as the autoregressive backend does.
        result.GeneratedTokens = ids.Length + (cut ? 1 : 0);
        result.ReachedTokenLimit = !cut;
        return tokenizer.Decode(ids);
    }

    /// <summary>Disposes the generator (and the model when owned); waits for a running generation to finish.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _gate.Wait();
        try
        {
            _generator?.Dispose();
            if (_ownsModel)
            {
                _model.Dispose();
            }
        }
        finally
        {
            _gate.Release();
        }
    }
}
