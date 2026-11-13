using System.CommandLine;
using Sharplight.Samples;

namespace Sharplight.AgentHost;

/// <summary>Declares the command line of the agent host and binds it to <see cref="AgentHostSettings"/>.</summary>
internal sealed class AgentHostOptionSet
{
    private readonly RuntimeOptionSet _runtime = new();

    private readonly Argument<string> _model = new("model")
    {
        Description = "Model to serve as an agent: a .gguf file, a .slm package or a Hugging Face checkpoint directory.",
    };

    private readonly Option<string?> _urls = new("--urls", "--listen")
    {
        HelpName = "url",
        Description = "Address to listen on (default: http://localhost:5000).",
    };

    /// <summary>Creates the root command; <paramref name="action"/> receives the validated settings.</summary>
    public RootCommand BuildCommand(Func<AgentHostSettings, CancellationToken, Task<int>> action)
    {
        var command = new RootCommand("Serves a local model as a Microsoft Agent Framework agent over OpenAI-compatible HTTP endpoints.");
        command.Arguments.Add(_model);
        command.Options.Add(_runtime.Cpu);
        command.Options.Add(_runtime.Device);
        command.Options.Add(_runtime.Context);
        command.Options.Add(_urls);
        command.SetAction(CommandRunner.Guard((parseResult, cancellationToken) =>
        {
            var settings = new AgentHostSettings(
                parseResult.GetValue(_model)!,
                parseResult.GetValue(_runtime.Cpu),
                parseResult.GetValue(_runtime.Device),
                parseResult.GetValue(_runtime.Context),
                parseResult.GetValue(_urls));
            return action(settings, cancellationToken);
        }));
        return command;
    }
}
