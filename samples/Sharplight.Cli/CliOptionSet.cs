using System.CommandLine;
using System.CommandLine.Parsing;
using System.Globalization;
using Sharplight.Samples;
using Sharplight.Vision;

namespace Sharplight.Cli;

/// <summary>Declares the command line of the <c>sharplight</c> tool and binds parsed values to <see cref="CliSettings"/>.</summary>
internal sealed class CliOptionSet
{
    private const int DefaultMaxNewTokens = 64;
    private const string DefaultPrompt = "The capital of France is";
    private const double BytesPerGiB = 1L << 30;

    private readonly RuntimeOptionSet _runtime = new();

    private readonly Argument<string?> _model = new("model")
    {
        Description = "Model to load: a .gguf file, a .slm package, or a Hugging Face checkpoint directory. Omit it to only list devices.",
        Arity = ArgumentArity.ZeroOrOne,
    };

    private readonly Option<bool> _devices = new("--devices")
    {
        Description = "List the available devices and their features before doing anything else.",
    };

    private readonly Option<string[]> _split = OptionFactory.CommaSeparatedList(
        "--split",
        "Comma-separated layer shares, one per device slot, e.g. 0.5,0.5. The same GPU is opened once per entry, which exercises the multi-device pipeline path.",
        "weights");

    private readonly Option<double?> _gpuMemory = new("--gpu-mem")
    {
        HelpName = "GiB",
        Description = "Hard memory cap per device slot in GiB (default: what the device reports as available).",
    };

    private readonly Option<string?> _mmproj = new("--mmproj")
    {
        HelpName = "file",
        Description = "Vision projector file for GGUF models (Hugging Face checkpoints contain their vision tower).",
    };

    private readonly Option<string[]> _images = new("--image")
    {
        HelpName = "file",
        Description = "Image to describe; repeatable. The prompt is formatted with the model's chat template.",
        AllowMultipleArgumentsPerToken = false,
        DefaultValueFactory = _ => [],
    };

    private readonly Option<int?> _imageMinTokens = OptionFactory.OptionalInteger("--image-min-tokens", "Minimum tokens per image; smaller images are upsized.", helpName: "tokens");

    private readonly Option<int?> _imageMaxTokens = OptionFactory.OptionalInteger("--image-max-tokens", "Maximum tokens per image; larger images are resized to fit.", helpName: "tokens");

    private readonly Option<string?> _save = new("--save")
    {
        HelpName = "file",
        Description = "Store the built model and this device's compiled kernels in one .slm package. When the model already is that package, the compiled target is appended to it.",
    };

    private readonly Option<int?> _bench = OptionFactory.OptionalInteger(
        "--bench",
        "Benchmark instead of generating: prefill N synthetic tokens, then 64 decode steps. Profile with SHARPLIGHT_VK_PROFILE=1.",
        helpName: "tokens");

    private readonly Option<string?> _logits = new("--logits")
    {
        HelpName = "file",
        Description = "Process the prompt, print the token ids and the top-5 logits, and store the last-token logits as raw float32 in this file.",
    };

    private readonly Option<string> _prompt = new("-p", "--prompt")
    {
        HelpName = "text",
        Description = "Prompt text; the two characters \\n are turned into a newline.",
        DefaultValueFactory = _ => DefaultPrompt,
    };

    private readonly Option<int> _maxNewTokens = OptionFactory.Integer("-n", "Maximum number of tokens to generate.", DefaultMaxNewTokens, minimum: 1, helpName: "count", "--max-new-tokens");

    private readonly Option<float> _temperature = OptionFactory.Single("--temp", "Sampling temperature; 0 selects greedy decoding.", 0f, minimum: 0f, helpName: "t");

    private readonly Option<bool> _chat = new("--chat")
    {
        Description = "Wrap the prompt as a user message and render it with the model's chat template.",
    };

    private readonly Option<bool> _think = new("--think")
    {
        Description = "Enable the model's thinking mode (only for image prompts, where the chat template is always used).",
    };

    /// <summary>Creates the root command; <paramref name="action"/> receives the validated settings.</summary>
    public RootCommand BuildCommand(Func<CliSettings, CancellationToken, Task<int>> action)
    {
        var command = new RootCommand("Loads a model and generates text, describes images or benchmarks it on a CPU or Vulkan device.");
        command.Arguments.Add(_model);
        command.Options.Add(_devices);
        _runtime.AddTo(command);
        command.Options.Add(_split);
        command.Options.Add(_gpuMemory);
        command.Options.Add(_mmproj);
        command.Options.Add(_images);
        command.Options.Add(_imageMinTokens);
        command.Options.Add(_imageMaxTokens);
        command.Options.Add(_save);
        command.Options.Add(_bench);
        command.Options.Add(_logits);
        command.Options.Add(_prompt);
        command.Options.Add(_maxNewTokens);
        command.Options.Add(_temperature);
        command.Options.Add(_chat);
        command.Options.Add(_think);
        command.Validators.Add(ValidateCombination);
        command.SetAction(CommandRunner.Guard((parseResult, cancellationToken) => action(Read(parseResult), cancellationToken)));
        return command;
    }

    private static bool HasAny(CommandResult result, params Option[] options)
    {
        return options.Any(option => result.GetResult(option) is not null);
    }

    private void ValidateCombination(CommandResult result)
    {
        if (HasAny(result, _bench) && HasAny(result, _logits))
        {
            result.AddError("--bench and --logits are mutually exclusive.");
        }

        if (HasAny(result, _bench, _logits, _save, _mmproj, _images) && result.GetValue(_model) is null)
        {
            result.AddError("A model is required for this command.");
        }
    }

    private CliSettings Read(ParseResult parseResult)
    {
        var modelPath = parseResult.GetValue(_model);
        bool listDevices = parseResult.GetValue(_devices) || modelPath is null;
        var splitWeights = ParseSplit(parseResult.GetValue(_split) ?? []);
        var imageTokens = ReadImageTokenLimits(parseResult);

        double? gpuMemoryGiB = parseResult.GetValue(_gpuMemory);
        long? gpuMemoryBytes = gpuMemoryGiB is { } gib ? (long)(gib * BytesPerGiB) : null;

        var generation = new GenerationSettings(
            parseResult.GetValue(_prompt)!.Replace("\\n", "\n"),
            parseResult.GetValue(_maxNewTokens),
            parseResult.GetValue(_temperature),
            parseResult.GetValue(_chat),
            parseResult.GetValue(_think));

        return new CliSettings(
            modelPath,
            listDevices,
            _runtime.Read(parseResult),
            splitWeights,
            gpuMemoryBytes,
            parseResult.GetValue(_mmproj),
            parseResult.GetValue(_images) ?? [],
            imageTokens,
            parseResult.GetValue(_save),
            parseResult.GetValue(_bench),
            parseResult.GetValue(_logits),
            generation);
    }

    private ImageTokenLimits? ReadImageTokenLimits(ParseResult parseResult)
    {
        int? min = parseResult.GetValue(_imageMinTokens);
        int? max = parseResult.GetValue(_imageMaxTokens);
        return min is null && max is null ? null : new ImageTokenLimits(min, max);
    }

    private static IReadOnlyList<double> ParseSplit(string[] entries)
    {
        if (entries.Length == 0)
        {
            return [1.0];
        }

        var weights = new List<double>(entries.Length);
        foreach (var entry in entries)
        {
            if (!double.TryParse(entry, NumberStyles.Float, CultureInfo.InvariantCulture, out double weight) || weight <= 0)
            {
                throw new SampleException($"--split expects positive numbers, got '{entry}'.");
            }

            weights.Add(weight);
        }

        return weights;
    }
}
