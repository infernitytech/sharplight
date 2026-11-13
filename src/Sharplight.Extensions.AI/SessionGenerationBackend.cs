using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Sharplight.Llm;
using Sharplight.Models;

namespace Sharplight.Extensions.AI;

/// <summary>
/// <see cref="ITextGenerationBackend"/> over a <see cref="LlmModel"/>: one <see cref="InferenceSession"/>, generations
/// are serialized (one at a time) and each runs from an empty context (no prefix reuse yet). The session
/// (and the vision encoder, on the first image request) are compiled lazily.
/// </summary>
public sealed class SessionGenerationBackend : ITextGenerationBackend
{
    private readonly LlmModel _model;
    private readonly bool _ownsModel;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private InferenceSession? _session;
    private VisionEncoder? _vision;
    private int _disposed;

    /// <summary>Wraps <paramref name="model"/>; <paramref name="ownsModel"/> disposes it with the backend.</summary>
    public SessionGenerationBackend(LlmModel model, bool ownsModel = false)
    {
        _model = model;
        _ownsModel = ownsModel;
        ContextLength = int.MaxValue; // refined once the session exists
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
    public bool SupportsImages => _model.SupportsImages;
    /// <inheritdoc/>
    public int ContextLength { get; private set; }
    /// <summary>Statistics of the last finished generation.</summary>
    public GenerationStats? LastStats { get; private set; }

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> GenerateAsync(TextGenerationRequest request, TextGenerationResult result,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            var producer = Task.Factory.StartNew(() => Produce(request, result, channel.Writer, cts.Token),
                CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
            try
            {
                await foreach (var piece in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
                {
                    yield return piece;
                }
            }
            finally
            {
                cts.Cancel();
                try
                {
                    await producer.ConfigureAwait(false);
                }
                catch
                {
                    // Already surfaced through the channel.
                }
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    private void Produce(TextGenerationRequest request, TextGenerationResult result, ChannelWriter<string> writer, CancellationToken ct)
    {
        try
        {
            var session = EnsureSession();
            var tokenizer = _model.Tokenizer;
            session.Reset();
            var options = request.Options;
            VisionEncoder? vision = null;
            if (request.Images.Count > 0)
            {
                if (!_model.SupportsImages)
                {
                    throw new NotSupportedException("The model has no vision tower (load it with a vision projector).");
                }

                vision = _vision ??= _model.CreateVisionEncoder();
            }
            var gen = new Generator(session, tokenizer, vision);
            IEnumerable<GeneratedToken> tokens;
            if (request.Images.Count > 0)
            {
                tokens = gen.Generate(request.Prompt, request.Images, options);
            }
            else
            {
                // The chat template emits BOS itself, so no special tokens are added here.
                var ids = tokenizer.Encode(request.Prompt, addSpecial: false, parseSpecial: true);
                if (ids.Length >= session.ContextLength)
                {
                    throw new InvalidOperationException($"The prompt ({ids.Length} tokens) does not fit the context ({session.ContextLength}).");
                }

                options = options with { MaxNewTokens = Math.Min(options.MaxNewTokens, session.ContextLength - ids.Length) };
                result.PromptTokens = ids.Length;
                tokens = gen.Generate(ids, options);
            }

            int visible = 0;
            try
            {
                foreach (var t in tokens)
                {
                    ct.ThrowIfCancellationRequested();
                    if (t.Id >= 0)
                    {
                        visible++;
                    }

                    if (t.Text.Length > 0)
                    {
                        writer.TryWrite(t.Text);
                    }
                }
            }
            finally
            {
                if (gen.LastStats is { } s)
                {
                    LastStats = s;
                    result.PromptTokens = s.PromptTokens;
                    result.GeneratedTokens = s.GeneratedTokens;
                    // The generator counts a terminating EOG token but does not yield it.
                    result.ReachedTokenLimit = s.GeneratedTokens >= options.MaxNewTokens && visible == s.GeneratedTokens;
                }
            }
            writer.TryComplete();
        }
        catch (Exception ex)
        {
            writer.TryComplete(ex);
        }
    }

    private InferenceSession EnsureSession()
    {
        if (_session is { } s)
        {
            return s;
        }

        s = _model.CreateSession(_model.SupportsImages ? WellKnown.DecodeMultimodalGraph : WellKnown.DecodeGraph);
        ContextLength = s.ContextLength;
        return _session = s;
    }

    /// <summary>Disposes the session and vision encoder (and the model when owned); waits for a running generation to finish.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _gate.Wait();
        try
        {
            _vision?.Dispose();
            _session?.Dispose();
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
