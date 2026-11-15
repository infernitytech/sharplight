using Xunit;

namespace Sharplight.Templating.Tests;

public sealed class TemplateSmokeTests
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
        foreach (var n in TestData.TemplateNames())
        {
            d.Add(n);
        }

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
