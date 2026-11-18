using System.CommandLine;
using System.CommandLine.Parsing;
using Sharplight.Samples;
using Sharplight.Samples.Hub;

namespace Sharplight.Cli;

/// <summary>Declares the command line of <c>sharplight serve</c> and binds parsed values to <see cref="ServeSettings"/>.</summary>
internal sealed class ServeOptionSet
{
    private const string DefaultListenUrl = "http://127.0.0.1:8080";
    /// <summary>Pulled models are precompiled for the default slot count.</summary>
    internal const int DefaultSlots = PrecompileTargets.ServeSlots;
    internal const int DefaultMaxQueue = 64;
    internal const int DefaultPrefillChunk = 0;
    internal const int DefaultSpeculationMaxBatch = 16;
    private const float DefaultTemperature = 0.7f;
    private const string ApiOpenAi = "openai";
    private const string ApiAnthropic = "anthropic";
    private const string ApiBoth = "both";

    private readonly RuntimeOptionSet _runtime = new();
    private readonly SampleRuntime _sampleRuntime;

    private readonly StoreOptionSet _store;
    private readonly PullOptionSet _pull = new(explicitPull: false);

    private readonly Argument<string[]> _positional = new("model")
    {
        Description = "Models to serve: stored models (owner/name[:tag] or short name), Hugging Face repositories (pulled when missing), "
            + ".slm/.gguf files or checkpoint directories.",
        Arity = ArgumentArity.ZeroOrMore,
    };

    private readonly Option<string[]> _models = new("--model")
    {
        HelpName = "NAME=MODEL[,key=value...]",
        Description = "Serve another model (file, stored or Hub model) under NAME; repeatable. Per-model keys: ctx, slots, batch, device (auto, cpu, an index, a device id such as vulkan:1, or several joined with +), mmproj, template, lazy, pin, idle (seconds), offload, kind (chat, embedding, reranker), task (a task kind such as speech-recognition).",
        DefaultValueFactory = _ => [],
    };

    private readonly Option<string?> _config = new("--config")
    {
        HelpName = "file",
        Description = "Hosting JSON file: host-wide settings, model defaults and a \"models\" list (docs/hosting.md). Options given on the command line win.",
    };

    private readonly Option<string?> _modelName = new("--model-name")
    {
        HelpName = "name",
        Description = "Model id reported for a single positional model (default: its file name, or the model name as given).",
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

    private readonly Option<int> _slots = OptionFactory.Integer("--slots", "Concurrent sequences batched per forward pass, each with up to --ctx tokens of context (from a paged pool shared by the slots, see --kv-tokens).", DefaultSlots, helpName: "count");

    private readonly Option<int> _maxQueue = OptionFactory.Integer("--max-queue", "Waiting requests per model before answering 429.", DefaultMaxQueue, helpName: "count");

    private readonly Option<int> _prefillChunk = OptionFactory.Integer("--prefill-chunk", "Prompt rows per step and sequence, interleaved with decoding (0: the step's rows, see --batch).", DefaultPrefillChunk, minimum: 0, helpName: "rows");

    private readonly Option<long?> _kvTokens = OptionFactory.Size("--kv-tokens",
        "Paged KV cache: one pool of this many tokens (K/M suffixes) shared by all --slots, each growing up to --ctx as needed; "
        + "admission and preemption follow free cache blocks, and prompts share cached blocks by content (prefix caching). "
        + "Default: --slots x --ctx tokens where the device runs the paged kernels; 0: every slot reserves --ctx tokens (contiguous cache).", helpName: "tokens");

    private readonly Option<bool> _allowCacheSalt = new("--allow-cache-salt")
    {
        Description = "Accept the cache_salt request field (vLLM's prefix-cache salt) from clients; it can only narrow prefix-cache sharing.",
    };

    private readonly Option<int?> _kvSnapshots = OptionFactory.OptionalInteger("--kv-snapshots",
        "Recurrent-state snapshots kept for prefix caching of models with recurrent state (gated delta nets, Mamba, RWKV); each costs one sequence's state. "
        + "Default: what fits in half the KV pool's memory or 512 MiB, at most two per slot; 0: no prefix caching for such models.", minimum: 0, helpName: "count");

    private readonly Option<bool> _noPrefixCache = new("--no-prefix-cache")
    {
        Description = "Compute every prompt in full instead of reusing the cached KV of earlier prompts with the same prefix.",
    };
    private readonly Option<int?> _stepTokens = OptionFactory.OptionalInteger("--step-tokens", "Token budget of one engine step (decode rows first, prompt chunks fill the rest; default: the model's batch size).", helpName: "tokens");

    private readonly Option<int> _draftTokens = OptionFactory.Integer("--speculate-ngram", "Prompt-lookup speculative decoding: draft up to this many tokens from earlier text (0 = off).", 0, minimum: 0, helpName: "tokens");
    private readonly Option<int> _specMaxBatch = OptionFactory.Integer("--speculate-max-batch", "Speculate only while at most this many sequences decode (larger batches are compute bound).", DefaultSpeculationMaxBatch, minimum: 1, helpName: "sequences");
    private readonly Option<int> _mtpTokens = OptionFactory.Integer("--speculate-mtp", "Multi-token-prediction speculative decoding with the model's own MTP head: draft this many tokens per step (0 = off; models without a head fall back to --speculate-ngram).", 0, minimum: 0, helpName: "tokens");

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

    private readonly Option<bool> _onDemand = new("--on-demand")
    {
        Description = "Also serve every model of the model store by name (ollama-style): it loads on its first request; "
            + "combine with --idle-timeout and --max-loaded / --vram-reserve to unload idle and least recently used models.",
    };

    public ServeOptionSet(StoreOptionSet store, SampleRuntime sampleRuntime)
    {
        _store = store;
        _sampleRuntime = sampleRuntime;
        _api.AcceptOnlyFromAmong(ApiOpenAi, ApiAnthropic, ApiBoth);
    }

    /// <summary>Adds the serve arguments and options to <paramref name="command"/>; <paramref name="action"/> receives the validated settings.</summary>
    public void Configure(Command command, Func<ServeSettings, CancellationToken, Task<int>> action)
    {
        command.Arguments.Add(_positional);
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
        command.Options.Add(_stepTokens);
        command.Options.Add(_kvTokens);
        command.Options.Add(_allowCacheSalt);
        command.Options.Add(_noPrefixCache);
        command.Options.Add(_kvSnapshots);
        command.Options.Add(_draftTokens);
        command.Options.Add(_mtpTokens);
        command.Options.Add(_specMaxBatch);
        command.Options.Add(_mmproj);
        command.Options.Add(_chatTemplate);
        command.Options.Add(_temperature);
        command.Options.Add(_vramReserve);
        command.Options.Add(_vramBudget);
        command.Options.Add(_maxLoaded);
        command.Options.Add(_lazy);
        command.Options.Add(_idleTimeout);
        command.Options.Add(_onDemand);
        _pull.AddTo(command);
        command.Validators.Add(ValidateModelSource);
        command.SetAction(HubGuard.Guard((parseResult, cancellationToken) => action(Read(parseResult), cancellationToken)));
    }

    private void ValidateModelSource(CommandResult result)
    {
        bool hasModel = result.GetValue(_positional) is { Length: > 0 }
            || result.GetValue(_models) is { Length: > 0 }
            || result.GetValue(_config) is not null
            || result.GetValue(_onDemand);
        if (!hasModel)
        {
            result.AddError("Give a model to serve: one or more positional <model>, --model NAME=MODEL, --config FILE, or --on-demand.");
            return;
        }

        if (result.GetValue(_modelName) is not null && result.GetValue(_positional) is not { Length: 1 })
        {
            result.AddError("--model-name names a single positional model.");
        }
    }

    private ServeSettings Read(ParseResult parseResult)
    {
        var runtime = _runtime.Read(parseResult);
        var pull = _pull.Read(parseResult, log => PrecompileTargets.Create(_sampleRuntime, true, runtime.UseCpu, runtime.ContextLength,
            runtime.BatchTokens, runtime.DisabledFeatures, log));
        return new ServeSettings(
            parseResult.GetValue(_positional) ?? [],
            parseResult.GetValue(_modelName),
            parseResult.GetValue(_models) ?? [],
            parseResult.GetValue(_config),
            runtime,
            parseResult.GetValue(_listen)!,
            ParseApi(parseResult.GetValue(_api)!),
            parseResult.GetValue(_apiKeys) ?? [],
            parseResult.GetValue(_cors) ?? [],
            parseResult.GetValue(_slots),
            parseResult.GetValue(_maxQueue),
            parseResult.GetValue(_prefillChunk),
            parseResult.GetValue(_stepTokens),
            parseResult.GetValue(_kvTokens),
            parseResult.GetValue(_draftTokens),
            parseResult.GetValue(_mtpTokens),
            parseResult.GetValue(_specMaxBatch),
            parseResult.GetValue(_mmproj),
            parseResult.GetValue(_chatTemplate),
            parseResult.GetValue(_temperature),
            parseResult.GetValue(_vramReserve),
            parseResult.GetValue(_vramBudget),
            parseResult.GetValue(_maxLoaded),
            parseResult.GetValue(_lazy),
            parseResult.GetValue(_idleTimeout),
            parseResult.GetValue(_onDemand),
            _store.ReadRoot(parseResult),
            pull,
            parseResult.GetValue(_allowCacheSalt),
            parseResult.GetValue(_noPrefixCache),
            parseResult.GetValue(_kvSnapshots));
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
