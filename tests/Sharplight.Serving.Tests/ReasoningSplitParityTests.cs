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

    /// <summary>An OpenAI harmony (gpt-oss) template: analysis and final channels instead of think markers (detected from the source).</summary>
    private const string Harmony =
        "{% for m in messages %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|end|>{% endfor %}" +
        "{% if add_generation_prompt %}<|start|>assistant\u0001{% endif %}{# <|channel|> #}";

    /// <summary>The tail of a Qwen3.5-style structured thinking block (indented lists that markdown would show as code).</summary>
    private const string StructuredThinking =
        "Thinking Process:\n\n1.  **Analyze:** the user says I am fast.\n2.  **Draft:**\n" +
        "    *   *Option 1 (Casual):* \"Haha, yeah, I try! What's on your mind?\"\n" +
        "    *   *Option 3 (Playful):* \"Speed is one of my features!\"\n\n" +
        "    *   Acknowledge the speed.\n    *   Offer assistance.\n";

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public static TheoryData<string, string, string, string> Outputs => new()
    {
        { nameof(OpensReasoning), "Let me think.\n</think>\n\nThe answer is 4.", "Let me think.", "The answer is 4." },
        { nameof(OpensReasoning), StructuredThinking + "</think>\n\nHaha, thanks! What can I do for you?", StructuredThinking.Trim(), "Haha, thanks! What can I do for you?" },
        { nameof(OpensReasoning), "\n\n</think>\n\nNo reasoning needed.", "", "No reasoning needed." },
        { nameof(OpensReasoning), "Cut off by the token limit while still thinking\n", "Cut off by the token limit while still thinking", "" },
        { nameof(LeavesReasoning), "<think>\nhmm\n</think>\n\nok", "hmm", "ok" },
        { nameof(LeavesReasoning), "  Hello, <b>world</b>!\n\n", "", "Hello, <b>world</b>!\n\n" },
        { nameof(Harmony), "<|channel|>analysis<|message|>The user is happy.<|end|><|start|>assistant<|channel|>final<|message|>Thanks!<|return|>", "The user is happy.", "Thanks!" },
    };

    [Theory]
    [MemberData(nameof(Outputs))]
    public async Task EveryPathSplitsReasoningAndContentTheSameWay(string templateName, string output, string reasoning, string content)
    {
        string template = templateName switch
        {
            nameof(OpensReasoning) => OpensReasoning,
            nameof(LeavesReasoning) => LeavesReasoning,
            _ => Harmony,
        };
        bool templateOpensReasoning = template == OpensReasoning;
        var expected = (reasoning, content);

        Assert.Equal(expected, ParseDirectly(output, templateOpensReasoning, ChatOutputParser.IsHarmonyTemplate(template)));

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

    private static (string Reasoning, string Content) ParseDirectly(string output, bool startsInReasoning, bool harmony)
    {
        var parser = new ChatOutputParser(new ChatOutputParserOptions { StartsInReasoning = startsInReasoning, Harmony = harmony });
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
