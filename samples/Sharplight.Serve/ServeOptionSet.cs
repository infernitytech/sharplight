using System.CommandLine;
using System.CommandLine.Parsing;
using Sharplight.Samples;

namespace Sharplight.Serve;

/// <summary>Declares the command line of <c>sharplight-serve</c> and binds parsed values to <see cref="ServeSettings"/>.</summary>
internal sealed class ServeOptionSet
{
    private const string DefaultListenUrl = "http://127.0.0.1:8080";
    private const int DefaultSlots = 4;
    private const int DefaultMaxQueue = 64;
    private const int DefaultPrefillChunk = 512;
    private const float DefaultTemperature = 0.7f;
    private const string ApiOpenAi = "openai";
    private const string ApiAnthropic = "anthropic";
    private const string ApiBoth = "both";

    private readonly RuntimeOptionSet _runtime = new();

    private readonly Argument<string?> _model = new("model")
    {
        Description = "A single model to serve: a .gguf file, a .slm package or a Hugging Face checkpoint directory.",
        Arity = ArgumentArity.ZeroOrOne,
    };

    private readonly Option<string[]> _models = new("--model")
    {
        HelpName = "NAME=PATH[,key=value...]",
        Description = "Serve another model; repeatable. Per-model keys: ctx, slots, batch, device (index or cpu), mmproj, template, lazy, pin, idle (seconds), offload.",
        DefaultValueFactory = _ => [],
    };

    private readonly Option<string?> _config = new("--config")
    {
        HelpName = "file",
        Description = "JSON file with host-wide settings and a \"models\" list (see the README).",
    };

    private readonly Option<string?> _modelName = new("--model-name")
    {
        HelpName = "name",
        Description = "Model id reported for the positional model (default: its file or directory name).",
    };

    private readonly Option<string> _listen = new("--listen", "--urls")
    {
        HelpName = "url",
        Description = "Address to listen on.",
        DefaultValueFactory = _ => DefaultListenUrl,
    };

    private readonly Option<string> _api = new("--api")
    {
        Description = "Which API to expose.",
        DefaultValueFactory = _ => ApiBoth,
    };

    private readonly Option<string[]> _apiKeys = new("--api-key")
    {
        HelpName = "key",
        Description = "Require this key (Authorization: Bearer or x-api-key); repeatable. Also protects /admin.",
        DefaultValueFactory = _ => [],
    };

    private readonly Option<string[]> _cors = new("--cors")
    {
        HelpName = "origin",
        Description = "Allow this CORS origin (\"*\" for any); repeatable.",
        DefaultValueFactory = _ => [],
    };

    private readonly Option<int> _slots = OptionFactory.Integer("--slots", "Concurrent sequences batched per forward pass, each with --ctx tokens of context.", DefaultSlots, helpName: "count");

    private readonly Option<int> _maxQueue = OptionFactory.Integer("--max-queue", "Waiting requests per model before answering 429.", DefaultMaxQueue, helpName: "count");

    private readonly Option<int> _prefillChunk = OptionFactory.Integer("--prefill-chunk", "Prompt rows per step and sequence, interleaved with decoding.", DefaultPrefillChunk, helpName: "rows");

    private readonly Option<string?> _mmproj = new("--mmproj")
    {
        HelpName = "file",
        Description = "Vision projector for the models without their own (images are not served yet).",
    };

    private readonly Option<string?> _chatTemplate = new("--chat-template")
    {
        HelpName = "file",
        Description = "Use this Jinja chat template instead of the models' own.",
    };

    private readonly Option<float> _temperature = OptionFactory.Single("--temp", "Default temperature for requests that give none.", DefaultTemperature, helpName: "t");

    private readonly Option<long?> _vramReserve = OptionFactory.Size("--vram-reserve", "Device memory kept free for other applications (K/M/G suffixes; default 1G).");

    private readonly Option<long?> _vramBudget = OptionFactory.Size("--vram-budget", "Never use more than this much device memory in total (default: what the driver reports as available).");

    private readonly Option<int?> _maxLoaded = OptionFactory.OptionalInteger("--max-loaded", "At most this many models resident at once; loading another evicts the least recently used idle one.", helpName: "count");

    private readonly Option<bool> _lazy = new("--lazy")
    {
        Description = "Load models on their first request instead of at startup.",
    };

    private readonly Option<double?> _idleTimeout = new("--idle-timeout")
    {
        HelpName = "seconds",
        Description = "Unload models that served nothing for this long; they reload on demand.",
    };

    public ServeOptionSet()
    {
        _api.AcceptOnlyFromAmong(ApiOpenAi, ApiAnthropic, ApiBoth);
    }

    /// <summary>Creates the root command; <paramref name="action"/> receives the validated settings.</summary>
    public RootCommand BuildCommand(Func<ServeSettings, CancellationToken, Task<int>> action)
    {
        var command = new RootCommand("Serves one or several models over the OpenAI and/or Anthropic HTTP API.");
        command.Arguments.Add(_model);
        command.Options.Add(_models);
        command.Options.Add(_config);
        command.Options.Add(_modelName);
        command.Options.Add(_listen);
        command.Options.Add(_api);
        command.Options.Add(_apiKeys);
        command.Options.Add(_cors);
        _runtime.AddTo(command);
        command.Options.Add(_slots);
        command.Options.Add(_maxQueue);
        command.Options.Add(_prefillChunk);
        command.Options.Add(_mmproj);
        command.Options.Add(_chatTemplate);
        command.Options.Add(_temperature);
        command.Options.Add(_vramReserve);
        command.Options.Add(_vramBudget);
        command.Options.Add(_maxLoaded);
        command.Options.Add(_lazy);
        command.Options.Add(_idleTimeout);
        command.Validators.Add(ValidateModelSource);
        command.SetAction(CommandRunner.Guard((parseResult, cancellationToken) => action(Read(parseResult), cancellationToken)));
        return command;
    }

    private void ValidateModelSource(CommandResult result)
    {
        bool hasModel = result.GetValue(_model) is not null
            || result.GetValue(_models) is { Length: > 0 }
            || result.GetValue(_config) is not null;
        if (!hasModel)
        {
            result.AddError("Give a model to serve: a positional <model>, one or more --model NAME=PATH, or --config FILE.");
        }
    }

    private ServeSettings Read(ParseResult parseResult)
    {
        return new ServeSettings(
            parseResult.GetValue(_model),
            parseResult.GetValue(_modelName),
            parseResult.GetValue(_models) ?? [],
            parseResult.GetValue(_config),
            _runtime.Read(parseResult),
            parseResult.GetValue(_listen)!,
            ParseApi(parseResult.GetValue(_api)!),
            parseResult.GetValue(_apiKeys) ?? [],
            parseResult.GetValue(_cors) ?? [],
            parseResult.GetValue(_slots),
            parseResult.GetValue(_maxQueue),
            parseResult.GetValue(_prefillChunk),
            parseResult.GetValue(_mmproj),
            parseResult.GetValue(_chatTemplate),
            parseResult.GetValue(_temperature),
            parseResult.GetValue(_vramReserve),
            parseResult.GetValue(_vramBudget),
            parseResult.GetValue(_maxLoaded),
            parseResult.GetValue(_lazy),
            parseResult.GetValue(_idleTimeout));
    }

    private static ApiSurface ParseApi(string value)
    {
        return value switch
        {
            ApiOpenAi => ApiSurface.OpenAi,
            ApiAnthropic => ApiSurface.Anthropic,
            _ => ApiSurface.Both,
        };
    }
}
