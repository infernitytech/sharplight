using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using Sharplight.Extensions.AI;
using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.AgentHost;

/// <summary>Registers the local model as an <see cref="IChatClient"/>, wraps it in an agent and maps the OpenAI endpoints.</summary>
internal static class AgentHostApplication
{
    private const string AgentName = "assistant";
    private const string AgentInstructions = "You are a helpful assistant.";

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

        builder.Services.AddSharplightChatClient(settings.ModelPath, _ => CreateLoadOptions(settings)).UseFunctionInvocation();
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

    private static LlmLoadOptions CreateLoadOptions(AgentHostSettings settings)
    {
        var provider = SampleRuntime.SelectProvider(settings.UseCpu);
        var device = provider.OpenDevice(SampleRuntime.GetDevice(provider, settings.DeviceIndex));
        return new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            ContextLength = settings.ContextLength,
            Log = Console.Error.WriteLine,
        };
    }
}
