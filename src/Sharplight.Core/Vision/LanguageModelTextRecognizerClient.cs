using System.Collections;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Tasks;
using Sharplight.Templating;

namespace Sharplight.Vision;

/// <summary>
/// The <see cref="ITextRecognizer"/> of a decoder-only vision-language model (GLM-OCR): the image and the task's prompt (<see cref="CanonicalKeys.RecognitionTasks"/>) go
/// into the user turn of the model's chat template, and the answer is generated greedily until the end of the turn; the text is the page's text (formulas as LaTeX, tables as markup,
/// as the task asks). The session is reset for every request. Requests are serialized.
/// </summary>
internal sealed class LanguageModelTextRecognizerClient : ITextRecognizer
{
    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly Dictionary<string, string> _tasks = new(StringComparer.Ordinal);
    private readonly ChatTemplate _template;
    private readonly object _gate = new();
    private VisionEncoder? _vision;
    private InferenceSession? _session;

    public LanguageModelTextRecognizerClient(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        _template = new ChatTemplate(_llm.ChatTemplate ?? throw new NotSupportedException("A text recognizer built on a language model needs the model's chat template."), "", "");
        foreach (var entry in ((IEnumerable)model.Definition.SourceMetadata[CanonicalKeys.RecognitionTasks]).Cast<object>().Select(o => o.ToString() ?? ""))
        {
            int at = entry.IndexOf('=', StringComparison.Ordinal);
            if (at > 0)
            {
                _tasks[entry[..at]] = entry[(at + 1)..];
            }
        }

        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public IReadOnlyList<string> Tasks => [.. _tasks.Keys];

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<TextRecognition> RecognizeAsync(ImageInput image, TextRecognitionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        image.Validate();
        options ??= new TextRecognitionOptions();
        cancellationToken.ThrowIfCancellationRequested();
        return Task.Run(() =>
        {
            lock (_gate)
            {
                return Recognize(image, options, cancellationToken);
            }
        }, cancellationToken);
    }

    public void Dispose()
    {
        _session?.Dispose();
        _vision?.Dispose();
    }

    private TextRecognition Recognize(ImageInput image, TextRecognitionOptions options, CancellationToken cancellationToken)
    {
        string task = options.Task ?? _tasks.Keys.First();
        if (!_tasks.TryGetValue(task, out var prompt))
        {
            throw new ArgumentException($"Unknown task '{task}'; the model's tasks are {string.Join(", ", _tasks.Keys)}.", nameof(options));
        }

        _vision ??= _llm.CreateVisionEncoder();
        _session ??= _llm.CreateSession(WellKnown.DecodeMultimodalGraph);
        _session.Reset();
        string text = _template.Render([new ChatMessage("user", Parts: [ChatContentPart.Image(), ChatContentPart.FromText(prompt)])], addGenerationPrompt: true,
            new Dictionary<string, object?>());
        var generator = new Generator(_session, _llm.Tokenizer, _vision);
        var generated = new System.Text.StringBuilder();
        var request = new GenerationOptions { MaxNewTokens = options.MaxTokens, Sampling = SamplingOptions.Greedy };
        foreach (var token in generator.Generate(text, [image], request))
        {
            cancellationToken.ThrowIfCancellationRequested();
            generated.Append(token.Text);
        }

        string result = generated.ToString().Trim();
        return new TextRecognition(result, [new TextRegion(result, null)]);
    }
}
