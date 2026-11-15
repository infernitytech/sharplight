using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.AgentHost;

/// <summary>Registers the local model as an <see cref="IChatClient"/>, wraps it in an agent and maps the OpenAI endpoints.</summary>
internal static class AgentHostApplication
{
    private const string AgentName = "assistant";
    private const string AgentInstructions = "You are a helpful assistant.";
    private const string CpuProviderName = "cpu";

    public static async Task<int> RunAsync(AgentHostSettings settings, CancellationToken cancellationToken)
    {
        if (!File.Exists(settings.ModelPath) && !Directory.Exists(settings.ModelPath))
        {
            throw new SampleException($"Model '{settings.ModelPath}' does not exist.");
        }

        var builder = WebApplication.CreateBuilder();
        if (settings.Urls is { } urls)
        {
            builder.WebHost.UseUrls(urls);
        }

        builder.Services.AddSharplight();
        builder.Services.AddSharplightChatClient((options, services) => ConfigureModel(options, services, settings)).UseFunctionInvocation();
        builder.AddOpenAIChatCompletions();
        builder.AddOpenAIResponses();
        builder.AddOpenAIConversations();
        var agent = builder.AddAIAgent(AgentName, AgentInstructions);

        var app = builder.Build();
        app.MapOpenAIChatCompletions(agent);
        app.MapOpenAIResponses(agent);
        app.MapOpenAIConversations();
        await app.RunAsync(cancellationToken);
        return ExitCodes.Success;
    }

    private static void ConfigureModel(LlmLoadOptions options, IServiceProvider services, AgentHostSettings settings)
    {
        var provider = SelectProvider(services.GetRequiredService<ExecutionProviderCatalog>(), settings.UseCpu);
        var device = provider.OpenDevice(SampleRuntime.GetDevice(provider, settings.DeviceIndex));
        var progress = new ConsoleLoadProgress();
        options.ModelPath = settings.ModelPath;
        options.Devices = [new DeviceSlot { Device = device }];
        options.ContextLength = settings.ContextLength;
        options.Log = progress.Log;
        options.Progress = progress;
    }

    private static IExecutionProvider SelectProvider(ExecutionProviderCatalog providers, bool cpu)
    {
        var match = providers.Discover().FirstOrDefault(d => d.Availability.IsAvailable && (d.Provider.Name == CpuProviderName) == cpu);
        return match?.Provider
            ?? throw new SampleException(cpu ? "The CPU provider is not available." : "No GPU provider is available; use --cpu to run on the CPU.");
    }
}
