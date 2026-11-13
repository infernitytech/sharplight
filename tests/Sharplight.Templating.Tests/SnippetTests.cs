using System.Text.Json;
using Sharplight.Templating;
using Xunit;

namespace Sharplight.Templating.Tests;

/// <summary>Snippets rendered by real Python jinja2 (see Goldens/gen_snippets.py); outputs must match exactly.</summary>
public class SnippetTests
{
    private static readonly Dictionary<string, JsonElement> Snippets = Load();

    private static Dictionary<string, JsonElement> Load() =>
        TestData.LoadJson("Goldens/snippets.json").EnumerateArray().ToDictionary(e => e.GetProperty("name").GetString()!, e => e);

    public static TheoryData<string> Names()
    {
        var d = new TheoryData<string>();
        foreach (var n in Snippets.Keys) d.Add(n);
        return d;
    }

    private static JinjaEnvironment MakeEnv(JsonElement options)
    {
        var env = new JinjaEnvironment
        {
            TrimBlocks = !options.TryGetProperty("trim", out var t) || t.GetBoolean(),
            LstripBlocks = !options.TryGetProperty("lstrip", out var l) || l.GetBoolean(),
            KeepTrailingNewline = options.TryGetProperty("keep_nl", out var k) && k.GetBoolean(),
            Clock = () => TestData.FixedNow,
        };
        return env;
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void Matches_python_jinja2(string name)
    {
        var s = Snippets[name];
        string template = s.GetProperty("template").GetString()!;
        var ctx = TestData.ToContext(s.GetProperty("context"));
        var env = MakeEnv(s.GetProperty("options"));

        if (s.TryGetProperty("error", out var err))
        {
            var ex = Assert.ThrowsAny<TemplateException>(() => env.Compile(template).Render(ctx));
            if (name == "err_raise")
            {
                Assert.IsType<TemplateRaisedException>(ex);
                Assert.Equal(err.GetString(), ex.Message);
            }
            return;
        }

        string expected = s.GetProperty("output").GetString()!;
        string actual = env.Compile(template).Render(ctx);
        Assert.Equal(expected, actual);
    }
}
