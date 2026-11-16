using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tokenizers;

namespace Sharplight.Vision;

/// <summary>
/// The built-in <see cref="ITextRecognizer"/> over an image-to-text encoder-decoder: the encoder component's <see cref="WellKnown.EncodeGraph"/> takes the patchified
/// image <c>patches</c> and returns the encoder rows, the decoder component's <see cref="WellKnown.DecodeGraph"/> is run token by token (greedy) from the start token
/// until the end token, with the encoder rows handed over on its first pass (it caches the cross-attention keys and values). Which model produced the graphs is irrelevant here.
/// <para>A prompted model (an encoder with a <c>tokens</c> input next to the image: Florence-2) takes the image as <c>pixels</c> and the task's prompt tokens; the model description lists
/// the tasks (<c>tasks</c>: task name to prompt text with an optional <c>{input}</c>) and the token ids around the prompt. The decoder's cross-attention is limited to the rows of the current
/// source, the decoder component's <c>forcedFirstToken</c> and <c>noRepeatNgram</c> shape the generation, and text with location tokens is returned as boxed regions.</para>
/// </summary>
internal sealed class TextRecognizerClient : ITextRecognizer
{
    private readonly TaskModel _model;
    private readonly CompiledModel _encoder;
    private readonly CompiledModel _decoder;
    private readonly ImageProcessor _processor;
    private readonly ITokenizer _tokenizer;
    private readonly Value? _patches;
    private readonly Value? _pixels;
    private readonly Value? _promptTokens;
    private readonly Value? _promptPositions;
    private readonly Value? _sourceSpans;
    private readonly Dictionary<string, string> _tasks = new(StringComparer.Ordinal);
    private readonly HashSet<string> _quadTasks = new(StringComparer.Ordinal);
    private readonly int _promptBos;
    private readonly int _promptEos;
    private readonly int _locationBins;
    private readonly int _forcedFirstToken;
    private readonly int _noRepeatNgram;
    private readonly Value _tokens;
    private readonly Value? _positions;
    private readonly Value _encoderOutput;
    private readonly Value? _outputRows;
    private readonly int _patchSize;
    private int _promptRows;
    private readonly int _sourceRows;
    private readonly int _width;
    private readonly int _startToken;
    private readonly int _endToken;
    private readonly int _contextLength;
    private readonly float[] _logits;
    private readonly float[] _encoded;
    private readonly RunParameters _run = new();
    private readonly object _gate = new();

    public TextRecognizerClient(TaskModel model)
    {
        _model = model;
        var extractor = model.Description.Component(ModelComponentRoles.FeatureExtractor);
        _processor = new ImageProcessor(ImageProcessorConfig.FromJson(extractor.Config[ImageProcessor.ComponentKey]));
        _tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        var decoderConfig = model.Description.Component(ModelComponentRoles.Decoder).Config;
        _startToken = decoderConfig["startToken"].GetInt32();
        _endToken = decoderConfig["endToken"].GetInt32();
        _encoder = model.CompileComponent(ModelComponentRoles.Encoder, WellKnown.EncodeGraph);
        _decoder = model.CompileComponent(ModelComponentRoles.Decoder, WellKnown.DecodeGraph);
        var output = _encoder.Graph.Outputs[WellKnown.EncoderOutput].Shape;
        _promptTokens = _encoder.Graph.TryInput(InputRole.TokenIds);
        if (_promptTokens is null)
        {
            _patches = _encoder.Graph.Inputs.First(v => v.Input!.Name == "patches");
            _patchSize = (int)Math.Round(Math.Sqrt(_patches.Shape[1].Fixed / 3.0));
            _sourceRows = (int)output[0].Fixed;
        }
        else
        {
            _pixels = _encoder.Graph.Inputs.First(v => v.Input!.Name == "pixels");
            _promptPositions = _encoder.Graph.Input(InputRole.Positions);
            _sourceRows = (int)model.Description.Component(ModelComponentRoles.Decoder).SymbolBounds![WellKnown.EncoderSymbol];
            var config = model.Description.Config;
            foreach (var task in config["tasks"].EnumerateObject())
            {
                _tasks[task.Name] = task.Value.GetString() ?? "";
            }

            foreach (var quad in config["quadTasks"].EnumerateArray())
            {
                _quadTasks.Add(quad.GetString() ?? "");
            }

            _promptBos = config["promptBos"].GetInt32();
            _promptEos = config["promptEos"].GetInt32();
            _locationBins = config["locationBins"].GetInt32();
        }

        _width = (int)output[1].Fixed;
        _encoded = new float[_sourceRows * _width];
        _forcedFirstToken = decoderConfig.TryGetValue("forcedFirstToken", out var forced) ? forced.GetInt32() : -1;
        _noRepeatNgram = decoderConfig.TryGetValue("noRepeatNgram", out var ngram) ? ngram.GetInt32() : 0;
        var graph = _decoder.Graph;
        _sourceSpans = graph.TryInput(InputRole.RowSpans);
        _tokens = graph.Input(InputRole.TokenIds);
        _positions = graph.TryInput(InputRole.Positions);
        _encoderOutput = graph.Input(InputRole.EncoderOutput);
        _outputRows = graph.TryInput(InputRole.OutputRows);
        _logits = new float[(int)graph.Outputs[WellKnown.LogitsOutput].Shape[-1].Fixed];
        _contextLength = Math.Min((int)_decoder.Options.ContextLength, decoderConfig["contextLength"].GetInt32());
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
        _encoder.Dispose();
        _decoder.Dispose();
    }

    private TextRecognition Recognize(ImageInput image, TextRecognitionOptions options, CancellationToken cancellationToken)
    {
        var processed = _processor.Process(image);
        if (_promptTokens is null)
        {
            _encoder.SetInput<float>(_patches!, processed.Patchify(_patchSize));
            _encoder.Run(new RunParameters());
            _encoder.GetOutput<float>(WellKnown.EncoderOutput, _encoded);
            _promptRows = _sourceRows;
        }
        else
        {
            EncodePrompted(processed, options);
        }

        var generated = new List<int>();
        var sequence = new List<int> { _startToken };
        int position = 0;
        int token = _startToken;
        int limit = Math.Min(options.MaxTokens, _contextLength - 1);
        while (generated.Count < limit)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var logits = Step(token, position);
            int next = position == 0 && _forcedFirstToken >= 0 ? _forcedFirstToken : Argmax(logits, sequence);
            position++;
            if (next == _endToken)
            {
                break;
            }

            generated.Add(next);
            sequence.Add(next);
            token = next;
        }

        string raw = _promptTokens is null ? "" : _tokenizer.Decode(generated.ToArray(), skipSpecial: false);
        if (LocationTextParser.HasLocations(raw))
        {
            var regions = LocationTextParser.Parse(raw, _locationBins, image.Width, image.Height, options.Task is { } task && _quadTasks.Contains(task));
            string cleaned = LocationTextParser.Clean(raw);
            return new TextRecognition(string.Join('\n', regions.Select(r => r.Text)), regions.Count > 0 ? regions : [new TextRegion(cleaned, null)]);
        }

        string text = _tokenizer.Decode(generated.ToArray(), skipSpecial: true).Trim();
        return new TextRecognition(text, [new TextRegion(text, null)]);
    }

    // The encoder over the image and the task's prompt: [bos] prompt [eos] tokens after the image rows.
    private void EncodePrompted(ProcessedImage processed, TextRecognitionOptions options)
    {
        string task = options.Task ?? _tasks.Keys.First();
        if (!_tasks.TryGetValue(task, out var template))
        {
            throw new ArgumentException($"Unknown task '{task}'; the model's tasks are {string.Join(", ", _tasks.Keys)}.", nameof(options));
        }

        if (template.Contains("{input}", StringComparison.Ordinal) && string.IsNullOrWhiteSpace(options.Input))
        {
            throw new ArgumentException($"The task {task} needs an input text.", nameof(options));
        }

        string prompt = template.Replace("{input}", options.Input?.Trim() ?? "", StringComparison.Ordinal);
        int[] tokens = [_promptBos, .. _tokenizer.Encode(prompt, addSpecial: false, parseSpecial: false), _promptEos];
        int imageRows = (int)_promptPositions!.Shape[0].Resolve(new Dictionary<string, long> { [WellKnown.TokenSymbol] = 0 });
        int rows = imageRows + tokens.Length;
        if (rows > _sourceRows)
        {
            throw new ArgumentException($"The prompt is too long: {tokens.Length} tokens and {imageRows} image rows exceed {_sourceRows} rows.", nameof(options));
        }

        _encoder.SetInput<float>(_pixels!, processed.Planes);
        _encoder.SetInput<int>(_promptTokens!, tokens);
        _encoder.SetInput<int>(_promptPositions!, [.. Enumerable.Range(0, rows)]);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = tokens.Length;
        _encoder.Run(run);
        _encoder.GetOutput<float>(WellKnown.EncoderOutput, _encoded.AsSpan(0, rows * _width));
        _promptRows = rows;
    }

    // One decoder step: the token at the position; the first step also carries the encoder rows.
    private float[] Step(int token, int position)
    {
        int sourceRows = position == 0 ? _promptRows : 0;
        _decoder.SetInput<int>(_tokens, [token]);
        if (_positions is not null)
        {
            _decoder.SetInput<int>(_positions, [position]);
        }

        _decoder.SetInput<float>(_encoderOutput, _encoded.AsSpan(0, sourceRows * _width));
        if (_outputRows is not null)
        {
            _decoder.SetInput<int>(_outputRows, [0]);
        }

        if (_sourceSpans is not null)
        {
            _decoder.SetInput<int>(_sourceSpans, [0, _promptRows]);
        }

        _run.Symbols[WellKnown.TokenSymbol] = 1;
        _run.Symbols[WellKnown.EncoderSymbol] = sourceRows;
        _run.Symbols["R"] = 1;
        _run.Scalars[WellKnown.PastScalar] = position;
        _run.Scalars[WellKnown.SourcePastScalar] = 0;
        _decoder.Run(_run);
        _decoder.GetOutput<float>(WellKnown.LogitsOutput, _logits);
        return _logits;
    }

    // The most likely token that does not repeat an n-gram of the sequence so far (when a size is set).
    private int Argmax(float[] values, List<int> sequence)
    {
        if (_noRepeatNgram > 1 && sequence.Count >= _noRepeatNgram - 1)
        {
            int prefix = _noRepeatNgram - 1;
            for (int start = 0; start + prefix < sequence.Count; start++)
            {
                bool match = true;
                for (int i = 0; i < prefix && match; i++)
                {
                    match = sequence[start + i] == sequence[sequence.Count - prefix + i];
                }

                if (match)
                {
                    values[sequence[start + prefix]] = float.NegativeInfinity;
                }
            }
        }

        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            best = values[i] > values[best] ? i : best;
        }

        return best;
    }
}
