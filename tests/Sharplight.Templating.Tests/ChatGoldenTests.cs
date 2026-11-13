using System.Text.Json;
using Sharplight.Templating;
using Xunit;

namespace Sharplight.Templating.Tests;

/// <summary>
/// Real chat templates (llama.cpp's models/templates plus the Qwen3.8 and Gemma 4 GGUF templates) rendered by Python jinja2
/// with the Hugging Face environment settings (Goldens/gen_chat_goldens.py). Output must match exactly.
/// </summary>
public class ChatGoldenTests
{
    private static readonly JsonElement Cases = TestData.LoadJson("Goldens/chat_cases.json");
    private static readonly JsonElement Goldens = TestData.LoadJson("Goldens/chat_goldens.json");

    public static TheoryData<string, string> AllGoldens()
    {
        var d = new TheoryData<string, string>();
        foreach (var t in Goldens.EnumerateObject())
            foreach (var c in t.Value.EnumerateObject())
                if (c.Name != "__compile_error__") d.Add(t.Name, c.Name);
        return d;
    }

    [Theory]
    [MemberData(nameof(AllGoldens))]
    public void Matches_python_jinja2(string template, string key)
    {
        var parts = key.Split('|');
        string caseName = parts[0];
        bool gen = parts[1] == "gen=1";
        string think = parts[2].Substring("think=".Length);
        var (messages, tools) = TestData.ToChat(Cases.GetProperty(caseName));
        Dictionary<string, object?>? extra = think == "n" ? null : new() { ["enable_thinking"] = think == "t" };

        var chat = new ChatTemplate(TestData.ReadTemplate(template), "<BOS>", "<EOS>", TestData.ChatEnv());
        var golden = Goldens.GetProperty(template).GetProperty(key);

        if (golden.TryGetProperty("error", out var err))
        {
            var ex = Assert.Throws<TemplateRaisedException>(() => chat.Render(messages, gen, extra, tools));
            Assert.Equal(err.GetString(), ex.Message);
        }
        else
        {
            Assert.Equal(golden.GetProperty("output").GetString(), chat.Render(messages, gen, extra, tools));
        }
    }
}

public class TemplateSmokeTests
{
    private static readonly HashSet<string> PythonAlsoFails =
    [
        "CohereForAI-c4ai-command-r-plus-tool_use",
        "NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use",
        "NousResearch-Hermes-3-Llama-3.1-8B-tool_use",
        "fireworks-ai-llama-3-firefunction-v2",
    ];

    public static TheoryData<string> Templates()
    {
        var d = new TheoryData<string>();
        foreach (var n in TestData.TemplateNames()) d.Add(n);
        return d;
    }

    [Theory]
    [MemberData(nameof(Templates))]
    public void Parses_and_renders_basic_conversation(string name)
    {
        var chat = new ChatTemplate(TestData.ReadTemplate(name), "<BOS>", "<EOS>", TestData.ChatEnv());
        var msgs = new[]
        {
            new ChatMessage("system", "You are helpful."),
            new ChatMessage("user", "Hi"),
            new ChatMessage("assistant", "Hello"),
            new ChatMessage("user", "Bye"),
        };
        try
        {
            var s = chat.Render(msgs, true);
            Assert.False(string.IsNullOrEmpty(s));
        }
        catch (TemplateRaisedException)
        {
            // templates may legitimately reject a conversation (e.g. role alternation rules)
        }
        catch (TemplateRuntimeException) when (PythonAlsoFails.Contains(name))
        {
            // these templates need extra variables (tools=[] / functions); Python jinja2 fails on them too
        }
    }
}
