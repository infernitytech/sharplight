using System.ComponentModel;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Sharplight.Extensions.AI.Tests;

public class PromptMappingTests
{
    private static readonly SharplightChatClientOptions Opts = new();

    [Fact]
    public void ChatOptionsMapToSampling()
    {
        var g = PromptBuilder.MapOptions(new ChatOptions { Temperature = 0.2f, TopP = 0.5f, TopK = 7, Seed = 42, MaxOutputTokens = 33, StopSequences = ["a", "b"] }, Opts);
        Assert.Equal(0.2f, g.Sampling.Temperature);
        Assert.Equal(0.5f, g.Sampling.TopP);
        Assert.Equal(7, g.Sampling.TopK);
        Assert.Equal(42, g.Sampling.Seed);
        Assert.Equal(33, g.MaxNewTokens);
        Assert.Equal(["a", "b"], g.StopStrings);
    }

    [Fact]
    public void DefaultsComeFromClientOptions()
    {
        var o = new SharplightChatClientOptions { DefaultMaxOutputTokens = 99, Sampling = new() { Temperature = 0.3f } };
        var g = PromptBuilder.MapOptions(null, o);
        Assert.Equal(99, g.MaxNewTokens);
        Assert.Equal(0.3f, g.Sampling.Temperature);
    }

    [Fact]
    public void SystemInstructionsAreMergedAndHoisted()
    {
        var msgs = PromptBuilder.MapMessages(
            [new(ChatRole.User, "hi"), new(ChatRole.System, "be brief"), new(ChatRole.Assistant, "hello")],
            new ChatOptions { Instructions = "Be nice." }, Opts, false, []);
        Assert.Equal(["system", "user", "assistant"], msgs.Select(m => m.Role));
        Assert.Equal("Be nice.\n\nbe brief", msgs[0].Content);
    }

    [Fact]
    public void FunctionCallsAndResults()
    {
        var args = new Dictionary<string, object?> { ["city"] = JsonDocument.Parse("\"Paris\"").RootElement, ["n"] = JsonDocument.Parse("2").RootElement };
        var msgs = PromptBuilder.MapMessages(
        [
            new(ChatRole.User, "weather?"),
            new(ChatRole.Assistant, [new FunctionCallContent("c1", "get_weather", args)]),
            new(ChatRole.Tool, [new FunctionResultContent("c1", "sunny")]),
        ], null, Opts, false, []);
        Assert.Equal(["user", "assistant", "tool"], msgs.Select(m => m.Role));
        Assert.Null(msgs[1].Content);
        var call = Assert.Single(msgs[1].ToolCalls!);
        Assert.Equal("Paris", call.Arguments!["city"]);
        Assert.Equal(2L, call.Arguments["n"]);
        Assert.Equal("sunny", msgs[2].Content);
        Assert.Equal("c1", msgs[2].ToolCallId);
    }

    [Fact]
    public void ImagesRequireVisionSupport()
    {
        var img = new DataContent(new byte[] { 1, 2, 3 }, "image/png");
        var ex = Assert.Throws<NotSupportedException>(() => PromptBuilder.MapMessages([new(ChatRole.User, [new TextContent("what?"), img])], null, Opts, false, []));
        Assert.Contains("vision", ex.Message);
    }

    [Fact]
    public void ImagesBecomePartsInOrder()
    {
        var o = new SharplightChatClientOptions { ImageDecoder = _ => new(1, 1, [1, 2, 3]) };
        var images = new List<Sharplight.Vision.ImageInput>();
        var msgs = PromptBuilder.MapMessages([new(ChatRole.User, [new DataContent(new byte[] { 0 }, "image/png"), new TextContent("what?")])], null, o, true, images);
        Assert.Single(images);
        Assert.Equal(["image", "text"], msgs[0].Parts!.Select(p => p.Type));
    }

    [Fact]
    public void ToolDefinitionsCarryNameDescriptionSchema()
    {
        var tool = AIFunctionFactory.Create(([Description("The city")] string city) => "x", "get_weather", "Gets weather");
        var def = Assert.Single(PromptBuilder.MapTools([tool]));
        Assert.Equal("get_weather", def.Name);
        Assert.Equal("Gets weather", def.Description);
        Assert.True(def.Parameters!.ContainsKey("properties"));
    }
}
