using System.Runtime.CompilerServices;
using System.Text;
using Sharplight.Tasks;

namespace Sharplight.Llm;

/// <summary>
/// The <see cref="ITextGenerator"/> of an <see cref="LlmModel"/>: one session (or, for masked-diffusion models, one diffusion
/// generator) compiled on first use, reset before every request, and the <see cref="Generator"/> sampling loop run on the thread
/// pool so callers do not block.
/// </summary>
internal sealed class TextGenerator : ITextGenerator
{
    private readonly LlmModel _model;
    private readonly string _graph;
    private readonly bool _ownsModel;
    private InferenceSession? _session;
    private DiffusionGenerator? _diffusion;
    private bool _disposed;

    /// <summary>Creates the generator; nothing is compiled until the first request.</summary>
    /// <param name="model">The model.</param>
    /// <param name="graph">The decoder graph for autoregressive models.</param>
    /// <param name="ownsModel">Whether disposing the generator disposes <paramref name="model"/>.</param>
    public TextGenerator(LlmModel model, string graph, bool ownsModel)
    {
        _model = model;
        _graph = graph;
        _ownsModel = ownsModel;
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        return serviceType.IsInstanceOfType(_model) ? _model : serviceType.IsInstanceOfType(this) ? this : null;
    }

    public async Task<TextGenerationResult> GenerateAsync(string prompt, GenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        var tokens = new List<GeneratedToken>();
        var text = new StringBuilder();
        GenerationStats? stats = null;
        await foreach (var token in Stream(prompt, options ?? new GenerationOptions(), s => stats = s, cancellationToken).ConfigureAwait(false))
        {
            tokens.Add(token);
            text.Append(token.Text);
        }

        return new TextGenerationResult(text.ToString(), tokens, stats);
    }

    public IAsyncEnumerable<GeneratedToken> GenerateStreamingAsync(string prompt, GenerationOptions? options = null, CancellationToken cancellationToken = default)
        => Stream(prompt, options ?? new GenerationOptions(), _ => { }, cancellationToken);

    private async IAsyncEnumerable<GeneratedToken> Stream(string prompt, GenerationOptions options, Action<GenerationStats?> reportStats,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_model.IsDiffusion)
        {
            _diffusion ??= await Task.Run(_model.CreateDiffusionGenerator, cancellationToken).ConfigureAwait(false);
            var generated = await Task.Run(() => _diffusion.Generate(prompt, options), cancellationToken).ConfigureAwait(false);
            reportStats(null);
            yield return new GeneratedToken(-1, generated);
            yield break;
        }

        _session ??= await Task.Run(() => _model.CreateSession(_graph), cancellationToken).ConfigureAwait(false);
        _session.Reset();
        var generator = new Generator(_session, _model.Tokenizer);
        using var tokens = generator.Generate(prompt, options).GetEnumerator();
        try
        {
            while (await Task.Run(tokens.MoveNext, cancellationToken).ConfigureAwait(false))
            {
                yield return tokens.Current;
            }
        }
        finally
        {
            reportStats(generator.LastStats);
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _session?.Dispose();
        _diffusion?.Dispose();
        if (_ownsModel)
        {
            _model.Dispose();
        }
    }
}
