using Sharplight.Templating;
using Xunit;

namespace Sharplight.Templating.Tests;

public class ChatTemplateTests
{
    private static ChatTemplate Qwen() =>
        new(TestData.ReadTemplate("Qwen3.8-27B"), "", "<|im_end|>", TestData.ChatEnv());

    private static readonly ChatMessage[] Basic =
    [
        new("system", "You are helpful."),
        new("user", "Hi"),
    ];

    [Fact]
    public void Qwen38_basic_prompt_thinking_default()
    {
        const string expected =
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\nYou are helpful.<|im_end|>\n" +
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        Assert.Equal(expected, Qwen().Render(Basic, true));
        Assert.Equal(expected, Qwen().Render(Basic, true, new Dictionary<string, object?> { ["enable_thinking"] = true }));
    }

    [Fact]
    public void Qwen38_basic_prompt_thinking_disabled()
    {
        const string expected =
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        Assert.Equal(expected, Qwen().Render(Basic, true, new Dictionary<string, object?> { ["enable_thinking"] = false }));
    }

    [Fact]
    public void Qwen38_without_generation_prompt()
    {
        var s = Qwen().Render(Basic, false, new Dictionary<string, object?> { ["enable_thinking"] = false });
        Assert.EndsWith("<|im_start|>user\nHi<|im_end|>\n", s);
    }

    [Fact]
    public void Qwen38_tools_and_tool_calls()
    {
        var tools = new[]
        {
            ToolDefinition.FromJson("""{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}"""),
        };
        var msgs = new ChatMessage[]
        {
            new("user", "Weather in Zürich?"),
            new("assistant", "", ToolCalls: [new ChatToolCall("c1", "get_weather", new Dictionary<string, object?> { ["city"] = "Zürich", ["days"] = 3, ["opts"] = new Dictionary<string, object?> { ["x"] = true } })]),
            new("tool", "sunny", ToolCallId: "c1"),
        };
        var s = Qwen().Render(msgs, true, new Dictionary<string, object?> { ["enable_thinking"] = false }, tools);
        Assert.Contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n</tools>", s);
        Assert.Contains("<tool_call>\n<function=get_weather>\n<parameter=city>\nZürich\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=opts>\n{\"x\": true}\n</parameter>\n</function>\n</tool_call><|im_end|>\n", s);
        Assert.Contains("<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n", s);
        Assert.EndsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n", s);
    }

    [Fact]
    public void Multimodal_parts_render_vision_tokens()
    {
        var msgs = new ChatMessage[]
        {
            new("user", Parts: [ChatContentPart.Image(), ChatContentPart.FromText("What is this?")]),
        };
        var s = Qwen().Render(msgs, true, new Dictionary<string, object?> { ["enable_thinking"] = false });
        Assert.Contains("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is this?<|im_end|>\n", s);
    }

    [Fact]
    public void Template_exceptions_surface_as_TemplateRaisedException()
    {
        var ex = Assert.Throws<TemplateRaisedException>(() => Qwen().Render([], true));
        Assert.Equal("No messages provided.", ex.Message);
    }

    [Fact]
    public void Bos_eos_and_extra_context_are_visible()
    {
        var t = new ChatTemplate("{{ bos_token }}|{{ eos_token }}|{{ x }}|{{ tools is none }}|{{ add_generation_prompt }}", "<s>", "</s>");
        Assert.Equal("<s>|</s>|7|True|False", t.Render([], false, new Dictionary<string, object?> { ["x"] = 7 }));
    }

    [Fact]
    public void Default_environment_uses_hf_whitespace_settings()
    {
        var t = new ChatTemplate("a\n  {% if true %}\nb\n  {% endif %}\nc\n", null, null);
        Assert.Equal("a\nb\nc", t.Render([], false));
        Assert.True(t.Environment.TrimBlocks);
        Assert.True(t.Environment.LstripBlocks);
    }

    [Fact]
    public void Role_dict_has_expected_keys()
    {
        var t = new ChatTemplate("{% for k in messages[0] %}{{ k }},{% endfor %}", null, null);
        var m = new ChatMessage("assistant", "x", ReasoningContent: "r", ToolCalls: [new ChatToolCall("i", "f")], Name: "n", ToolCallId: "t");
        Assert.Equal("role,content,reasoning_content,tool_calls,name,tool_call_id,", t.Render([m], false));
    }
}
