using Xunit;

namespace Sharplight.Templating.Tests;

public class EngineTests
{
    private static string R(string src, Dictionary<string, object?>? ctx = null, Action<JinjaEnvironment>? configure = null)
    {
        var env = new JinjaEnvironment { Clock = () => TestData.FixedNow };
        configure?.Invoke(env);
        return env.Compile(src).Render(ctx);
    }

    [Fact]
    public void Undefined_is_lenient_by_default()
    {
        Assert.Equal("[][][]", R("[{{ a }}][{{ a.b.c }}][{{ a['x'].y }}]"));
        Assert.Equal("F", R("{{ 'T' if a.b else 'F' }}"));
        Assert.Equal("", R("{% for i in a.b %}x{% endfor %}"));
        Assert.Equal("yes", R("{{ 'yes' if a.b is undefined }}"));
    }

    [Fact]
    public void Strict_undefined_throws_on_use_but_not_on_tests()
    {
        void S(string src) => Assert.Throws<TemplateRuntimeException>(() => R(src, null, e => e.StrictUndefined = true));
        S("{{ a }}");
        S("{{ a.b }}");
        S("{% if a %}x{% endif %}");
        S("{% for i in a %}x{% endfor %}");
        S("{{ {'k': 1}.nope }}");
        Assert.Equal("d", R("{{ a|default('d') }}", null, e => e.StrictUndefined = true));
        Assert.Equal("True", R("{{ a is undefined }}", null, e => e.StrictUndefined = true));
    }

    [Fact]
    public void Errors_carry_line_and_column()
    {
        var ex = Assert.Throws<TemplateRuntimeException>(() => R("line1\nline2 {{ 1 + 'a' }}"));
        Assert.Equal(2, ex.Line);
        Assert.True(ex.Column >= 7);
        Assert.Contains("line 2", ex.Message);

        var syn = Assert.Throws<TemplateSyntaxException>(() => R("a\n\n{% if x %}\nb"));
        Assert.Contains("endif", syn.Message);

        var syn2 = Assert.Throws<TemplateSyntaxException>(() => R("{{ 1 +\n }}"));
        Assert.Equal(2, syn2.Line);

        var raised = Assert.Throws<TemplateRaisedException>(() => R("x{{ raise_exception('nope') }}"));
        Assert.Equal("nope", raised.Message);
        Assert.Equal(1, raised.Line);
    }

    [Fact]
    public void Unknown_filter_and_test_have_names_in_message()
    {
        Assert.Contains("nofilter", Assert.Throws<TemplateRuntimeException>(() => R("{{ 1|nofilter }}")).Message);
        Assert.Contains("notest", Assert.Throws<TemplateRuntimeException>(() => R("{{ 1 is notest }}")).Message);
    }

    [Fact]
    public void Generation_tags_pass_through()
    {
        Assert.Equal("ab", R("a{% generation %}b{% endgeneration %}"));
    }

    [Fact]
    public void Custom_globals_filters_and_tests()
    {
        var env = new JinjaEnvironment();
        env.Globals["answer"] = 42L;
        env.Globals["twice"] = new NativeFunction("twice", a => Ops2.Twice(a.Positional[0]));
        env.Filters["shout"] = (e, v, a) => Ops2.Str(v).ToUpperInvariant() + "!";
        env.Tests["answerish"] = (e, v, a) => v is long l && l == 42;
        Assert.Equal("42 8 HI! True", env.Compile("{{ answer }} {{ twice(4) }} {{ 'hi'|shout }} {{ answer is answerish }}").Render());
    }

    private static class Ops2
    {
        public static object? Twice(object? v) => (long)v! * 2;
        public static string Str(object? v) => v?.ToString() ?? "";
    }

    [Fact]
    public void Host_values_are_normalized()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["i"] = 5,
            ["f"] = 1.5f,
            ["l"] = new[] { 1, 2, 3 },
            ["d"] = new Dictionary<string, object?> { ["x"] = new List<string> { "a", "b" } },
            ["c"] = 'z',
        };
        Assert.Equal("5 1.5 3 b z 2", R("{{ i }} {{ f }} {{ l|length }} {{ d.x[1] }} {{ c }} {{ l[1] }}", ctx));
    }

    [Fact]
    public void Render_is_repeatable_and_does_not_mutate_template()
    {
        var t = new JinjaEnvironment().Compile("{% set ns = namespace(n=0) %}{% for i in range(3) %}{% set ns.n = ns.n + i %}{% endfor %}{{ ns.n }}{{ x }}");
        Assert.Equal("3a", t.Render(new Dictionary<string, object?> { ["x"] = "a" }));
        Assert.Equal("3b", t.Render(new Dictionary<string, object?> { ["x"] = "b" }));
    }

    [Fact]
    public void Concurrent_renders_are_safe()
    {
        var t = new JinjaEnvironment().Compile("{% for i in range(50) %}{{ i * n }},{% endfor %}");
        var expected = Enumerable.Range(0, 8).Select(n => t.Render(new Dictionary<string, object?> { ["n"] = n })).ToArray();
        Parallel.For(0, 200, k =>
        {
            int n = k % 8;
            Assert.Equal(expected[n], t.Render(new Dictionary<string, object?> { ["n"] = n }));
        });
    }

    [Fact]
    public void Tojson_matches_python_dumps_format()
    {
        var d = new JinjaDict { ["a"] = 1L, ["b"] = new List<object?> { 1.0, "é\n", null, true }, ["c"] = new JinjaDict() };
        Assert.Equal("{\"a\": 1, \"b\": [1.0, \"é\\n\", null, true], \"c\": {}}", JinjaJson.Dumps(d));
        Assert.Equal("{\n  \"a\": 1,\n  \"b\": [\n    1.0,\n    \"\\u00e9\\n\",\n    null,\n    true\n  ],\n  \"c\": {}\n}", JinjaJson.Dumps(d, ensureAscii: true, indent: "  "));
        Assert.Equal("{\"a\":1,\"b\":[1.0,\"é\\n\",null,true],\"c\":{}}", JinjaJson.Dumps(d, itemSeparator: ",", keySeparator: ":"));
    }

    [Fact]
    public void Json_roundtrip_through_from_json()
    {
        Assert.Equal("3 b", R("{% set o = '{\"a\": [1, 2, 3], \"b\": \"b\"}'|from_json %}{{ o.a|length }} {{ o.b }}"));
    }

    [Fact]
    public void Unsupported_tags_fail_clearly()
    {
        var ex = Assert.Throws<TemplateSyntaxException>(() => R("{% include 'x' %}"));
        Assert.Contains("include", ex.Message);
    }

    [Fact]
    public void Float_repr_matches_python()
    {
        Assert.Equal("0.1 1e-05 1e+16 123456789.123 1.0 -0.0 0.30000000000000004 1.7976931348623157e+308", R("{{ 0.1 }} {{ 0.00001 }} {{ 1e16 }} {{ 123456789.123 }} {{ 1.0 }} {{ -0.0 }} {{ 0.1 + 0.2 }} {{ 1.7976931348623157e308 }}"));
    }

    [Fact]
    public void Strftime_now_uses_clock()
    {
        Assert.Equal("15 Jan 2026|2026-01-15 10:30:45", R("{{ strftime_now('%d %b %Y') }}|{{ strftime_now('%Y-%m-%d %H:%M:%S') }}"));
    }

    [Fact]
    public void Deep_recursion_does_not_crash_for_reasonable_depth()
    {
        Assert.Equal("5050", R("{% macro s(n) %}{% if n == 0 %}0{% else %}{{ n + s(n - 1)|int }}{% endif %}{% endmacro %}{{ s(100) }}"));
    }
}
