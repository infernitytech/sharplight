using System.Diagnostics;
using System.Globalization;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Hosting;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Samples;

namespace Sharplight.Cli;

/// <summary>
/// The terminal chat of <c>run</c>: hosts the model with Sharplight.Hosting and chats through the serving stack (chat template, reasoning and
/// tool-call parsing, prefix caching between turns) and streams each answer. Reasoning is shown dimmed on a terminal and
/// left out when stdout is redirected.
/// </summary>
internal sealed class ChatRepl(SampleRuntime sampleRuntime, RunSettings settings)
{
    private const string UserPrompt = ">>> ";
    private const string ExitCommand = "/bye";
    private const string ClearCommand = "/clear";
    private const string Dim = "\e[2m";
    private const string Normal = "\e[0m";
    private const int ChatSlots = 2;

    private readonly bool _interactiveOutput = !Console.IsOutputRedirected;
    private readonly List<ChatTurn> _history = [];

    public async Task RunAsync(string modelPath, CancellationToken cancellationToken)
    {
        var runtime = settings.Runtime;
        var services = new ServiceCollection();
        sampleRuntime.AddSharedServices(services);
        services.AddSharplightHosting(options =>
        {
            // One model the user asked for: no memory kept back, the host only refuses what does not fit at all.
            options.Reserve = 0;
            options.DisabledFeatures.AddRange(runtime.DisabledFeatures);
            options.Models.Add(new HostedModelOptions
            {
                Name = settings.Model,
                Path = modelPath,
                Device = runtime.UseCpu ? DeviceSpecifications.Cpu : runtime.DeviceIndex.ToString(CultureInfo.InvariantCulture),
                ContextLength = runtime.ContextLength,
                BatchTokens = runtime.BatchTokens,
                KvCacheType = runtime.KvCacheType.Name,
                CpuOffload = runtime.CpuOffload,
                // A multi-sequence decoder can roll its context back to the part the next turn's prompt shares (the template rewrites
                // the end of earlier answers, e.g. drops their reasoning), which the single-slot decoder cannot; a paged pool of one
                // context keeps the memory at one slot's.
                Slots = ChatSlots,
                KvCacheTokens = runtime.ContextLength,
                Lazy = true,
            });
            options.Defaults.ReducedPrecision = runtime.ReducedPrecision;
            options.ConfigureLoad = (_, load) =>
            {
                var progress = new ConsoleLoadProgress();
                load.Log = progress.Log;
                load.Progress = progress;
            };
            options.ConfigureChat = (_, chat) => chat.Service = chat.Service with { DefaultMaxTokens = settings.MaxTokens };
        });

        await using var provider = services.BuildServiceProvider();
        var manager = provider.GetRequiredService<IModelHostManager>();
        using var lease = await manager.Host.AcquireAsync(settings.Model, ModelKind.Chat, cancellationToken);
        var service = ((IChatModel)lease.As<Llm.Serving.Hosting.LoadedModel>()).Service;
        if (!service.HasChatTemplate)
        {
            throw new SampleException("The model has no chat template; it cannot be used for chat.");
        }

        StartConversation();
        if (settings.Prompt is { } prompt)
        {
            await AnswerAsync(service, prompt, cancellationToken);
            return;
        }

        await ChatAsync(service, cancellationToken);
    }

    private async Task ChatAsync(ModelService service, CancellationToken cancellationToken)
    {
        bool interactiveInput = !Console.IsInputRedirected;
        if (interactiveInput)
        {
            Console.Error.WriteLine($"Chatting with {settings.Model}. Ctrl+C stops an answer, {ClearCommand} resets the conversation, "
                + $"{ExitCommand} or Ctrl+D quits.");
        }

        while (!cancellationToken.IsCancellationRequested)
        {
            if (interactiveInput)
            {
                Console.Write(UserPrompt);
            }

            var line = Console.ReadLine();
            if (line is null || line.Trim() == ExitCommand)
            {
                return;
            }

            if (line.Trim() == ClearCommand)
            {
                StartConversation();
                continue;
            }

            if (line.Trim().Length > 0)
            {
                await AnswerAsync(service, line, cancellationToken);
            }
        }
    }

    private void StartConversation()
    {
        _history.Clear();
        if (settings.SystemPrompt is { } system)
        {
            _history.Add(new ChatTurn("system", system));
        }
    }

    private async Task AnswerAsync(ModelService service, string message, CancellationToken cancellationToken)
    {
        _history.Add(new ChatTurn("user", message));
        var request = new ChatRequest
        {
            Messages = [.. _history],
            MaxTokens = settings.MaxTokens,
            Sampling = new SamplingParams { Temperature = settings.Temperature },
            EnableThinking = settings.Think ? null : false,
        };

        var content = new StringBuilder();
        var reasoning = new StringBuilder();
        var stopwatch = Stopwatch.StartNew();
        GenerationUsage? usage = null;
        string? error = null;
        // Ctrl+C stops this answer only; the conversation goes on with what was generated so far.
        using var interrupt = ConsoleInterrupt.Begin(cancellationToken);
        try
        {
            await foreach (var chunk in service.StartChat(request, interrupt.Token).ReadAllAsync(interrupt.Token))
            {
                if (chunk.Reasoning is { } thought)
                {
                    reasoning.Append(thought);
                    if (_interactiveOutput)
                    {
                        Console.Write(Dim + thought + Normal);
                    }
                }

                if (chunk.Content is { } text)
                {
                    content.Append(text);
                    Console.Write(text);
                }

                usage = chunk.Usage ?? usage;
                error = chunk.Error ?? error;
            }
        }
        catch (OperationCanceledException) when (interrupt.Interrupted && !cancellationToken.IsCancellationRequested)
        {
            usage = null;
        }

        Console.WriteLine();
        _history.Add(new ChatTurn("assistant", content.ToString(), Reasoning: reasoning.Length > 0 ? reasoning.ToString() : null));
        if (interrupt.Interrupted)
        {
            Console.Error.WriteLine("[interrupted]");
        }
        else if (error is not null)
        {
            Console.Error.WriteLine($"error: {error}");
        }
        else if (usage is { } counts)
        {
            Console.Error.WriteLine($"[{counts.PromptTokens} prompt tokens ({counts.CachedTokens} cached), {counts.CompletionTokens} generated, "
                + $"{counts.CompletionTokens / Math.Max(stopwatch.Elapsed.TotalSeconds, 1e-3):F1} tok/s overall]");
        }
    }
}
