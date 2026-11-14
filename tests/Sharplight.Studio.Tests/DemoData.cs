using Sharplight.Hub;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Server;
using Sharplight.Studio.Tests.Fakes;

namespace Sharplight.Studio.Tests;

/// <summary>Plausible content for the window tests and the documentation screenshots.</summary>
internal static class DemoData
{
    public const string Answer = """
        Here is a small C# helper that streams a chat answer from a local Sharplight server:

        ```csharp
        using var client = new HttpClient { BaseAddress = new Uri("http://127.0.0.1:8080") };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "qwen3-0.6b",
            messages = new[] { new { role = "user", content = "Hello!" } },
        });
        Console.WriteLine(await response.Content.ReadAsStringAsync());
        ```

        | Option | Meaning |
        | --- | --- |
        | `model` | Which **resident** model answers |
        | `stream` | Server-sent events instead of one response |

        - Requests route by their `model` field
        - Two chats on *different* models run side by side
        """;

    public static async Task FillAsync(TestServices services)
    {
        var qwen = services.Runtime.Add(new FakeModel("qwen3-0.6b", ModelCapabilities.Chat | ModelCapabilities.Reasoning | ModelCapabilities.Tools)
        {
            ResidentBytes = 1_400_000_000,
            LastUsed = DateTimeOffset.UtcNow.AddMinutes(-2),
        });
        services.Runtime.Add(new FakeModel("gemma-3-4b-it", ModelCapabilities.Chat | ModelCapabilities.Vision)
        {
            ResidentBytes = 4_100_000_000,
            IsPinned = false,
            LastUsed = DateTimeOffset.UtcNow.AddMinutes(-11),
        });
        services.Runtime.Add(new FakeModel("qwen3-next-80b-a3b", ModelCapabilities.Chat | ModelCapabilities.Reasoning | ModelCapabilities.DraftHead)
        {
            State = ResidentState.Loading,
            ResidentBytes = 0,
            EstimatedBytes = 9_800_000_000,
            Device = FakeRuntime.Cpu,
            LoadStatus = new LoadStatus(0.46, "Compiling kernels 112 / 240 (64 cached)"),
        });

        var start = DateTimeOffset.UtcNow.AddMinutes(-5);
        var user = new ChatMessageRecord { Role = ChatRole.User, Content = "How do I call the local server from C#?", CreatedAt = start };
        var answer = new ChatMessageRecord
        {
            ParentId = user.Id,
            Role = ChatRole.Assistant,
            ModelId = qwen.Id,
            Content = Answer,
            Reasoning = "The user wants a C# example. Use HttpClient against the OpenAI-compatible endpoint.",
            CreatedAt = start.AddSeconds(4),
            Stats = new MessageStats { PromptTokens = 38, CompletionTokens = 212, PrefillMs = 41, DecodeMs = 1630, QueueMs = 2, DraftTokens = 60, AcceptedDraftTokens = 41 },
        };
        await services.Conversations.SaveAsync(new Conversation
        {
            Title = "Calling the server from C#",
            ModelId = qwen.Id,
            Messages = [user, answer],
            CurrentLeafId = answer.Id,
        });
        await services.Conversations.SaveAsync(new Conversation { Title = "Summarize the release notes", UpdatedAt = start.AddHours(-3) });
        await services.Conversations.SaveAsync(new Conversation { Title = "Vision: what is on this chart?", UpdatedAt = start.AddDays(-1) });

        var random = new Random(7);
        var now = DateTimeOffset.UtcNow;
        for (int i = 0; i < 60; i++)
        {
            var model = i % 3 == 0 ? "gemma-3-4b-it" : "qwen3-0.6b";
            double speed = model == "qwen3-0.6b" ? 180 + random.Next(40) : 62 + random.Next(12);
            int completion = 80 + random.Next(400);
            int prompt = 40 + random.Next(1500);
            await services.Stats.AddAsync(new RequestRecord
            {
                Model = model,
                Timestamp = now.AddMinutes(-i * 22 - random.Next(10)),
                Source = i % 4 == 0 ? RequestSource.Api : RequestSource.Chat,
                Device = "Test GPU",
                Stats = new MessageStats
                {
                    PromptTokens = prompt,
                    CompletionTokens = completion,
                    PrefillMs = prompt / (model == "qwen3-0.6b" ? 6.5 : 2.1),
                    DecodeMs = completion / speed * 1000,
                    QueueMs = random.Next(5),
                    DraftTokens = completion / 3,
                    AcceptedDraftTokens = completion / 5,
                },
            });
        }

        services.Library.Entries.Add(new LibraryEntry
        {
            Id = "Qwen/Qwen3-0.6B-GGUF:Q8_0",
            DisplayName = "Qwen/Qwen3-0.6B-GGUF:Q8_0",
            Path = "/models/qwen3/model.slm",
            Architecture = "qwen3",
            Quantization = "Q8_0",
            Format = "GGUF",
            SizeBytes = 639_000_000,
            Capabilities = ModelCapabilities.Chat | ModelCapabilities.Reasoning | ModelCapabilities.Tools,
            TrainedContextLength = 40960,
            AddedAt = now.AddDays(-3),
            Precompiled = ["vulkan"],
            Manifest = new ModelManifest { Repository = "Qwen/Qwen3-0.6B-GGUF", Tag = "Q8_0", Alias = "qwen3-0.6b", Commit = "c0ffee" },
        });
        services.Library.Entries.Add(new LibraryEntry
        {
            Id = "google/gemma-3-4b-it-GGUF:Q4_K_M",
            DisplayName = "google/gemma-3-4b-it-GGUF:Q4_K_M",
            Path = "/models/gemma3/model.slm",
            Architecture = "gemma3",
            Quantization = "Q4_K_M",
            Format = "GGUF",
            SizeBytes = 2_490_000_000,
            Capabilities = ModelCapabilities.Chat | ModelCapabilities.Vision,
            TrainedContextLength = 131072,
            AddedAt = now.AddDays(-1),
            Manifest = new ModelManifest { Repository = "google/gemma-3-4b-it-GGUF", Tag = "Q4_K_M", Alias = "gemma-3-4b-it", Commit = "beef" },
        });
        services.Library.Entries.Add(new LibraryEntry
        {
            Id = "/data/models/phi-4-mini.gguf",
            DisplayName = "phi-4-mini",
            Path = "/data/models/phi-4-mini.gguf",
            Architecture = "phi3",
            Quantization = "Q4_K_M",
            Format = "GGUF",
            SizeBytes = 2_320_000_000,
            Capabilities = ModelCapabilities.Chat | ModelCapabilities.Tools,
            TrainedContextLength = 131072,
        });

        services.Hub.Results.AddRange(
        [
            new HubModelSummary("Qwen/Qwen3-8B-GGUF", 412_000, 1820, false, "gguf", "text-generation", 8_190_000_000, [ModelFormat.Gguf], ["Q4_K_M", "Q5_K_M", "Q8_0"]),
            new HubModelSummary("unsloth/gemma-3-12b-it-GGUF", 238_000, 960, false, "gguf", "image-text-to-text", 11_800_000_000, [ModelFormat.Gguf], ["Q4_K_M", "Q6_K", "mmproj-F16"]),
            new HubModelSummary("meta-llama/Llama-3.2-3B-Instruct", 1_920_000, 2100, true, "transformers", "text-generation", 3_210_000_000, [ModelFormat.Safetensors], []),
            new HubModelSummary("onnx-community/Qwen3-0.6B-ONNX", 18_400, 75, false, "transformers.js", "text-generation", 600_000_000, [ModelFormat.Onnx], ["model_q4f16", "model_fp16"]),
        ]);

        services.Settings.StoreRoot = "~/.local/share/sharplight/models";
        services.Hub.FirstUpdate = new PullUpdate(PullPhase.Downloading, "Pulling Qwen/Qwen3-8B-GGUF:Q4_K_M", Received: 2_140_000_000, Total: 5_030_000_000,
            File: "Qwen3-8B-Q4_K_M.gguf", FileReceived: 2_140_000_000, FileSize: 5_030_000_000);
        await services.Server.StartAsync(new ServerSettings(), CancellationToken.None);
        var log = new[] { ("POST", "/v1/chat/completions", 200, 1830.0), ("GET", "/v1/models", 200, 2.1), ("POST", "/v1/messages", 200, 920.4), ("POST", "/v1/chat/completions", 404, 1.2) };
        foreach (var (method, path, status, ms) in log)
        {
            services.Server.Log(new HttpLogEntry(now.AddSeconds(-log.Length), method, path, status, ms, "127.0.0.1"));
        }
    }
}
