using Microsoft.Extensions.AI;
using Sharplight.Extensions.AI;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Every chat path splits the same model output the same way, because they share one <see cref="ChatOutputParser"/>: the parser on
/// its own (fed one character at a time), the serving engine behind the OpenAI / Anthropic endpoints and Studio
/// (<see cref="ServingChatClient"/> over <see cref="ModelService"/>, one token per byte) and the in-process
/// <see cref="SharplightChatClient"/>, streaming and non-streaming.
/// </summary>
public sealed class ReasoningSplitParityTests
{
    /// <summary>A template whose generation prompt opens a reasoning block (Qwen3 / Qwen3.5 / DeepSeek-R1 with thinking on).</summary>
    private const string OpensReasoning =
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}" +
        "{% if add_generation_prompt %}<|im_start|>assistant\n\u0001<think>\n{% endif %}";

    /// <summary>A template whose generation prompt leaves reasoning to the model.</summary>
    private const string LeavesReasoning =
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}" +
        "{% if add_generation_prompt %}<|im_start|>assistant\n\u0001{% endif %}";

    /// <summary>The tail of a Qwen3.5-style structured thinking block (indented lists that markdown would show as code).</summary>
    private const string StructuredThinking =
        "Thinking Process:\n\n1.  **Analyze:** the user says I am fast.\n2.  **Draft:**\n" +
        "    *   *Option 1 (Casual):* \"Haha, yeah, I try! What's on your mind?\"\n" +
        "    *   *Option 3 (Playful):* \"Speed is one of my features!\"\n\n" +
        "    *   Acknowledge the speed.\n    *   Offer assistance.\n";

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public static TheoryData<bool, string, string, string> Outputs => new()
    {
        { true, "Let me think.\n</think>\n\nThe answer is 4.", "Let me think.", "The answer is 4." },
        { true, StructuredThinking + "</think>\n\nHaha, thanks! What can I do for you?", StructuredThinking.Trim(), "Haha, thanks! What can I do for you?" },
        { true, "\n\n</think>\n\nNo reasoning needed.", "", "No reasoning needed." },
        { true, "Cut off by the token limit while still thinking\n", "Cut off by the token limit while still thinking", "" },
        { false, "<think>\nhmm\n</think>\n\nok", "hmm", "ok" },
        { false, "  Hello, <b>world</b>!\n\n", "", "Hello, <b>world</b>!\n\n" },
    };

    [Theory]
    [MemberData(nameof(Outputs))]
    public async Task EveryPathSplitsReasoningAndContentTheSameWay(bool templateOpensReasoning, string output, string reasoning, string content)
    {
        string template = templateOpensReasoning ? OpensReasoning : LeavesReasoning;
        var expected = (reasoning, content);

        Assert.Equal(expected, ParseDirectly(output, templateOpensReasoning));

        var (engine, service) = CreateService(template, output);
        using (engine)
        {
            using var serving = new ServingChatClient(service);
            Assert.Equal(expected, await StreamAsync(serving));
            Assert.Equal(expected, Split(await serving.GetResponseAsync("q", cancellationToken: Ct)));
        }

        using var backend = new ScriptedTextBackend(template, output);
        using var client = new SharplightChatClient(backend);
        Assert.Equal(expected, await StreamAsync(client));
        Assert.Equal(templateOpensReasoning, ChatOutputParser.PromptEndsInReasoning(backend.LastPrompt!));
        Assert.Equal(expected, Split(await client.GetResponseAsync("q", cancellationToken: Ct)));
    }

    private static (string Reasoning, string Content) ParseDirectly(string output, bool startsInReasoning)
    {
        var parser = new ChatOutputParser(new ChatOutputParserOptions { StartsInReasoning = startsInReasoning });
        var pieces = new List<ChatOutputPiece>();
        foreach (char c in output)
        {
            pieces.AddRange(parser.Push(c.ToString()));
        }

        pieces.AddRange(parser.Complete());
        return (Join(pieces, ChatOutputPieceKind.Reasoning), Join(pieces, ChatOutputPieceKind.Content));
    }

    private static string Join(IEnumerable<ChatOutputPiece> pieces, ChatOutputPieceKind kind)
        => string.Concat(pieces.Where(piece => piece.Kind == kind).Select(piece => piece.Text));

    private static (InferenceEngine Engine, ModelService Service) CreateService(string template, string output)
    {
        var decoder = new FakeBatchDecoder(1);
        // The scripted reply is indexed from the prompt's marker: an opened reasoning block in the prompt is part of the script.
        decoder.Script(template == OpensReasoning ? "<think>\n" + output : output);
        var engine = new InferenceEngine(decoder, new FakeTokenizer(), new EngineOptions());
        return (engine, new ModelService(engine, template, new ModelServiceOptions { ModelName = "fake", DefaultMaxTokens = 1024 }));
    }

    private static async Task<(string Reasoning, string Content)> StreamAsync(IChatClient client)
    {
        var updates = new List<ChatResponseUpdate>();
        await foreach (var update in client.GetStreamingResponseAsync("q", cancellationToken: Ct))
        {
            updates.Add(update);
        }

        return Split(updates.SelectMany(update => update.Contents));
    }

    private static (string Reasoning, string Content) Split(ChatResponse response) => Split(response.Messages.SelectMany(message => message.Contents));

    private static (string Reasoning, string Content) Split(IEnumerable<AIContent> contents)
    {
        var list = contents.ToList();
        return (string.Concat(list.OfType<TextReasoningContent>().Select(r => r.Text)), string.Concat(list.OfType<TextContent>().Select(t => t.Text)));
    }
}
