using System.CommandLine;
using Sharplight.Samples;

namespace Sharplight.Server;

/// <summary><c>run &lt;model&gt;</c>: an interactive chat with one model, pulling it first when it is not stored.</summary>
internal sealed class RunCommand(StoreOptionSet store)
{
    private const float DefaultTemperature = 0.7f;
    private const int DefaultMaxTokens = 2048;

    private readonly Argument<string> _model = new("model")
    {
        Description = "A stored model (owner/name[:tag] or its short name), a Hugging Face repository to pull, or a .slm/.gguf file or checkpoint directory.",
    };

    private readonly RuntimeOptionSet _runtime = new();
    private readonly PullOptionSet _pull = new(explicitPull: false);

    private readonly Option<float> _temperature = OptionFactory.Single("--temp", "Sampling temperature (0 = greedy).", DefaultTemperature, helpName: "t");

    private readonly Option<int> _maxTokens = OptionFactory.Integer("--max-tokens", "Maximum tokens per answer.", DefaultMaxTokens, helpName: "tokens");

    private readonly Option<string?> _system = new("--system") { HelpName = "text", Description = "System message that starts the conversation." };

    private readonly Option<string?> _prompt = new("--prompt", "-p") { HelpName = "text", Description = "Answer this prompt and exit instead of chatting." };

    private readonly Option<bool> _noThink = new("--no-think") { Description = "Ask reasoning models to answer without thinking first (where the chat template supports it)." };

    public Command Build()
    {
        var command = new Command("run", "Chats with a model in the terminal (/clear resets the conversation, /bye or Ctrl+D quits). "
            + "Models that are not stored are pulled first. Lines piped to stdin are sent as successive messages.");
        command.Arguments.Add(_model);
        _runtime.AddTo(command);
        _pull.AddTo(command);
        command.Options.Add(_temperature);
        command.Options.Add(_maxTokens);
        command.Options.Add(_system);
        command.Options.Add(_prompt);
        command.Options.Add(_noThink);
        command.SetAction(HubGuard.Guard(async (parseResult, cancellationToken) =>
        {
            var settings = Read(parseResult);
            using var hub = new HubContext(store.ReadRoot(parseResult));
            var path = await new ModelResolver(hub, settings.Pull).ResolveAsync(settings.Model, cancellationToken);
            await new ChatRepl(settings).RunAsync(path, cancellationToken);
            return ExitCodes.Success;
        }));
        return command;
    }

    private RunSettings Read(ParseResult parseResult)
    {
        var runtime = _runtime.Read(parseResult);
        var pull = _pull.Read(parseResult, log => PrecompileTargets.Create(true, runtime.UseCpu, runtime.DeviceIndex, runtime.ContextLength,
            runtime.BatchTokens, runtime.DisabledFeatures, log));
        return new RunSettings(
            parseResult.GetValue(_model)!,
            runtime,
            pull,
            parseResult.GetValue(_temperature),
            parseResult.GetValue(_maxTokens),
            parseResult.GetValue(_system),
            parseResult.GetValue(_prompt),
            !parseResult.GetValue(_noThink));
    }
}
