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
/// </summary>
internal sealed class TextRecognizerClient : ITextRecognizer
{
    private readonly TaskModel _model;
    private readonly CompiledModel _encoder;
    private readonly CompiledModel _decoder;
    private readonly ImageProcessor _processor;
    private readonly ITokenizer _tokenizer;
    private readonly Value _patches;
    private readonly Value _tokens;
    private readonly Value? _positions;
    private readonly Value _encoderOutput;
    private readonly Value? _outputRows;
    private readonly int _patchSize;
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
        _patches = _encoder.Graph.Inputs.First(v => v.Input!.Name == "patches");
        _patchSize = (int)Math.Round(Math.Sqrt(_patches.Shape[1].Fixed / 3.0));
        var output = _encoder.Graph.Outputs[WellKnown.EncoderOutput].Shape;
        _sourceRows = (int)output[0].Fixed;
        _width = (int)output[1].Fixed;
        _encoded = new float[_sourceRows * _width];
        var graph = _decoder.Graph;
        _tokens = graph.Input(InputRole.TokenIds);
        _positions = graph.TryInput(InputRole.Positions);
        _encoderOutput = graph.Input(InputRole.EncoderOutput);
        _outputRows = graph.TryInput(InputRole.OutputRows);
        _logits = new float[(int)graph.Outputs[WellKnown.LogitsOutput].Shape[-1].Fixed];
        _contextLength = Math.Min((int)_decoder.Options.ContextLength, decoderConfig["contextLength"].GetInt32());
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

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
        _encoder.SetInput<float>(_patches, processed.Patchify(_patchSize));
        _encoder.Run(new RunParameters());
        _encoder.GetOutput<float>(WellKnown.EncoderOutput, _encoded);

        var generated = new List<int>();
        int position = 0;
        int token = _startToken;
        int limit = Math.Min(options.MaxTokens, _contextLength - 1);
        while (generated.Count < limit)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int next = Argmax(Step(token, position));
            position++;
            if (next == _endToken)
            {
                break;
            }

            generated.Add(next);
            token = next;
        }

        string text = _tokenizer.Decode(generated.ToArray(), skipSpecial: true).Trim();
        return new TextRecognition(text, [new TextRegion(text, null)]);
    }

    // One decoder step: the token at the position; the first step also carries the encoder rows.
    private float[] Step(int token, int position)
    {
        int sourceRows = position == 0 ? _sourceRows : 0;
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

        _run.Symbols[WellKnown.TokenSymbol] = 1;
        _run.Symbols[WellKnown.EncoderSymbol] = sourceRows;
        _run.Symbols["R"] = 1;
        _run.Scalars[WellKnown.PastScalar] = position;
        _run.Scalars[WellKnown.SourcePastScalar] = 0;
        _decoder.Run(_run);
        _decoder.GetOutput<float>(WellKnown.LogitsOutput, _logits);
        return _logits;
    }

    private static int Argmax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            best = values[i] > values[best] ? i : best;
        }

        return best;
    }
}
