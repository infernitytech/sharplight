using System.Text;
using System.Text.Json;
using Sharplight.Llm.Structured;

namespace Sharplight.Serving.Tests;

/// <summary>Grammar front ends (regex, GBNF, JSON schema), the byte automaton and the trie-based token masks.</summary>
public class GrammarTests
{
    private static bool Matches(Grammar g, string text)
    {
        var rt = new GrammarRuntime(g);
        var s = rt.Advance(rt.Initial, Encoding.UTF8.GetBytes(text));
        return s != rt.Dead && s.Accepting;
    }

    private static bool IsPrefix(Grammar g, string text)
    {
        var rt = new GrammarRuntime(g);
        return rt.Advance(rt.Initial, Encoding.UTF8.GetBytes(text)) != rt.Dead;
    }

    [Theory]
    [InlineData(@"\d{3}-[a-z]+", "123-abc", true)]
    [InlineData(@"\d{3}-[a-z]+", "12-abc", false)]
    [InlineData(@"\d{3}-[a-z]+", "123-", false)]
    [InlineData(@"(ab|cd)*e?", "", true)]
    [InlineData(@"(ab|cd)*e?", "abcdabe", true)]
    [InlineData(@"(ab|cd)*e?", "abc", false)]
    [InlineData(@"^[α-ω]+$", "αβγ", true)]
    [InlineData(@"[α-ω]+", "αβA", false)]
    [InlineData(@"a.c", "a€c", true)]
    [InlineData(@"a.c", "a\nc", false)]
    [InlineData(@"[^a-z]{2}", "Ä1", true)]
    [InlineData(@"[^a-z]{2}", "Äb", false)]
    [InlineData(@"(?:x|y){2,3}", "xyx", true)]
    [InlineData(@"(?:x|y){2,3}", "xyxy", false)]
    [InlineData(@"\w+@\w+\.com", "me@site.com", true)]
    [InlineData(@"[😀-😂]", "😁", true)]
    public void Regex_MatchesWholeText(string pattern, string text, bool expected)
    {
        Assert.Equal(expected, Matches(Grammar.FromRegex(pattern), text));
    }

    [Fact]
    public void Regex_RejectsUnsupportedFeatures()
    {
        Assert.Throws<FormatException>(() => Grammar.FromRegex(@"(a)\1"));
        Assert.Throws<FormatException>(() => Grammar.FromRegex(@"(?=a)b"));
        Assert.Throws<FormatException>(() => Grammar.FromRegex(@"[a-"));
    }

    [Fact]
    public void Gbnf_ParsesLlamaCppStyleGrammar()
    {
        const string Text = """
            # arithmetic
            root  ::= expr
            expr  ::= term (("+" | "-") term)*
            term  ::= factor
                      ("*" factor)*
            factor ::= num | "(" expr ")"
            num   ::= [0-9]+
            """;
        var g = Grammar.FromGbnf(Text);
        Assert.True(Matches(g, "1+2*(3-4)"));
        Assert.True(Matches(g, "42"));
        Assert.False(Matches(g, "1+"));
        Assert.True(IsPrefix(g, "1+"));
        Assert.False(IsPrefix(g, "1+)"));
    }

    [Fact]
    public void Gbnf_EscapesClassesAndBoundedRepeats()
    {
        var g = Grammar.FromGbnf("""root ::= "\x41\n" [^"\\]{1,3} "é" .""");
        Assert.True(Matches(g, "A\nabcé!"));
        Assert.True(Matches(g, "A\nxyzé😀"));
        Assert.False(Matches(g, "A\nwxyzé!"));
    }

    [Fact]
    public void Grammar_RejectsLeftRecursionAndUndefinedRules()
    {
        Assert.Throws<ArgumentException>(() => Grammar.FromGbnf("root ::= root \"a\" | \"b\""));
        Assert.Throws<ArgumentException>(() => Grammar.FromGbnf("root ::= x? root \"a\" | \"b\"\nx ::= \"y\""));
        Assert.Throws<ArgumentException>(() => Grammar.FromGbnf("root ::= missing"));
    }

    private const string PersonSchema = """
        {
          "type": "object",
          "properties": {
            "name": { "type": "string", "minLength": 1, "maxLength": 20 },
            "age": { "type": "integer", "minimum": 0 },
            "email": { "type": "string", "format": "email" },
            "role": { "enum": ["admin", "user"] },
            "tags": { "type": "array", "items": { "type": "string" }, "maxItems": 3 }
          },
          "required": ["name", "role"],
          "additionalProperties": false
        }
        """;

    [Theory]
    [InlineData("""{"name":"Ann","role":"user"}""", true)]
    [InlineData("""{ "name": "Ann", "age": 41, "role": "admin", "tags": ["a", "b"] }""", true)]
    [InlineData("""{"name":"Ann","age":41,"email":"a@b.c","role":"user","tags":[]}""", true)]
    [InlineData("""{"name":"Ann"}""", false)]
    [InlineData("""{"role":"user","name":"Ann"}""", false)]
    [InlineData("""{"name":"Ann","role":"guest"}""", false)]
    [InlineData("""{"name":"","role":"user"}""", false)]
    [InlineData("""{"name":"Ann","age":-1,"role":"user"}""", false)]
    [InlineData("""{"name":"Ann","role":"user","tags":["a","b","c","d"]}""", false)]
    [InlineData("""{"name":"Ann","role":"user","extra":1}""", false)]
    [InlineData("""{"name":"Ann",,"role":"user"}""", false)]
    public void JsonSchema_Object(string json, bool valid)
    {
        Assert.Equal(valid, Matches(Grammar.FromJsonSchema(PersonSchema), json));
    }

    [Fact]
    public void JsonSchema_RecursiveRefsNullableAndAnyOf()
    {
        const string Schema = """
            {
              "$defs": { "node": { "type": "object", "properties": {
                  "v": { "type": ["integer", "null"] },
                  "kids": { "type": "array", "items": { "$ref": "#/$defs/node" } } }, "required": ["v"] } },
              "anyOf": [ { "$ref": "#/$defs/node" }, { "type": "boolean" } ]
            }
            """;
        var g = Grammar.FromJsonSchema(Schema);
        Assert.True(Matches(g, """{"v":1,"kids":[{"v":null},{"v":2,"kids":[]}]}"""));
        Assert.True(Matches(g, "true"));
        Assert.False(Matches(g, """{"v":1,"kids":[{"kids":[]}]}"""));
    }

    [Fact]
    public void JsonSchema_PatternFormatsAndTuples()
    {
        var g = Grammar.FromJsonSchema("""
            { "type": "array", "prefixItems": [ { "type": "string", "pattern": "^[A-Z]{2}\\d$" }, { "type": "string", "format": "date" } ],
              "items": false }
            """);
        Assert.True(Matches(g, """["AB1", "2024-02-29"]"""));
        Assert.False(Matches(g, """["AB1", "2024-13-01"]"""));
        Assert.False(Matches(g, """["AB1", "2024-02-29", "x"]"""));
    }

    [Fact]
    public void JsonObject_AcceptsArbitraryObjects()
    {
        var g = Grammar.JsonObject;
        Assert.True(Matches(g, """{"a":[1,2.5e3,-0.1,{"b":null}],"c":"x\"\\é é","d":true}"""));
        Assert.True(Matches(g, "\n{ }\n"));
        Assert.False(Matches(g, "[1]"));
        Assert.False(Matches(g, """{"a":01}"""));
        Assert.False(Matches(g, "{\"a\":\"\u0001\"}"));
    }

    private static readonly string[] Words =
        ["{\"", "\":", "\": ", "\",", "name", "na", "me\"", "\"}", "true", "tr", "ue", "12", "123", "  ", "\n", "},", "ab", "αβ", "\"name\": \""];

    [Fact]
    public void TokenMask_EqualsBruteForceAlongRandomWalk()
    {
        var tok = new VocabTokenizer(Words);
        var g = Grammar.FromJsonSchema("""
            { "type": "object", "properties": { "name": { "type": "string" }, "ok": { "type": "boolean" }, "n": { "type": "integer" } },
              "required": ["name"] }
            """);
        var constraint = new GrammarConstraint(g, tok);
        var rng = new Random(7);
        for (int run = 0; run < 20; run++)
        {
            var matcher = constraint.CreateMatcher();
            var reference = new GrammarRuntime(g);
            var state = reference.Initial;
            var mask = new TokenMask(tok.VocabSize);
            for (int step = 0; step < 40 && !matcher.IsTerminated; step++)
            {
                matcher.FillAllowed(mask);
                var allowed = new List<int>();
                for (int id = 0; id < tok.VocabSize; id++)
                {
                    bool expected = id != tok.Eos && reference.Advance(state, tok.TokenBytes(id)) != reference.Dead;
                    Assert.Equal(expected, mask.IsAllowed(id));
                    if (expected)
                    {
                        allowed.Add(id);
                    }
                }

                Assert.Equal(state.Accepting, matcher.IsComplete);
                if (allowed.Count == 0)
                {
                    break;
                }

                int pick = allowed[rng.Next(allowed.Count)];
                Assert.True(matcher.Accept(pick));
                state = reference.Advance(state, tok.TokenBytes(pick));
            }
        }
    }

    [Fact]
    public void Matcher_RejectsDisallowedToken_AndRollsBack()
    {
        var tok = new VocabTokenizer(Words);
        var matcher = new GrammarConstraint(Grammar.FromRegex("(ab)+c"), tok).CreateMatcher();
        int ab = tok.TokenId("ab")!.Value;
        Assert.False(matcher.Accept('c'));
        Assert.True(matcher.Accept(ab));
        Assert.True(matcher.Accept(ab));
        Assert.False(matcher.IsComplete);
        Assert.True(matcher.Accept('c'));
        Assert.True(matcher.IsComplete);
        Assert.True(matcher.IsTerminated);
        matcher.Rollback(2);
        var mask = new TokenMask(tok.VocabSize);
        matcher.FillAllowed(mask);
        Assert.True(mask.IsAllowed(ab));
        Assert.True(mask.IsAllowed('c'));
        matcher.Rollback(1);
        matcher.FillAllowed(mask);
        Assert.True(mask.IsAllowed('a'));
        Assert.False(mask.IsAllowed('c'));
        Assert.Throws<ArgumentOutOfRangeException>(() => matcher.Rollback(5));
    }

    [Fact]
    public void Trigger_LeavesTextFreeUntilItAppears()
    {
        var tok = new VocabTokenizer(Words, special: ["<call>"]);
        int call = tok.TokenId("<call>")!.Value;
        var constraint = new GrammarConstraint(Grammar.FromRegex("[0-9]+"), tok, new GrammarConstraintOptions { Trigger = "</think>", TriggerOptional = false });
        var matcher = constraint.CreateMatcher();
        var mask = new TokenMask(tok.VocabSize);
        matcher.FillAllowed(mask);
        Assert.True(mask.IsAllowed('x'));
        Assert.False(matcher.IsComplete);
        foreach (char c in "hm </thi")
        {
            Assert.True(matcher.Accept(c));
        }

        Assert.True(matcher.Accept('n'));
        Assert.True(matcher.Accept('k'));
        Assert.True(matcher.Accept('>'));
        matcher.FillAllowed(mask);
        Assert.False(mask.IsAllowed('x'));
        Assert.True(mask.IsAllowed('7'));
        Assert.False(mask.IsAllowed(call));
        Assert.True(matcher.Accept('7'));
        Assert.True(matcher.IsComplete);
        matcher.Rollback(2);
        matcher.FillAllowed(mask);
        Assert.True(mask.IsAllowed('x'));
    }

    [Fact]
    public void SpecialTokens_AreAllowedOnlyWhenNamed()
    {
        var tok = new VocabTokenizer(Words, special: ["<call>", "<other>"]);
        int call = tok.TokenId("<call>")!.Value;
        int other = tok.TokenId("<other>")!.Value;
        var g = Grammar.FromGbnf("""root ::= "<call>" [0-9] | "<other>" """);
        var mask = new TokenMask(tok.VocabSize);
        var named = new GrammarConstraint(g, tok, new GrammarConstraintOptions { SpecialTokens = ["<call>"] }).CreateMatcher();
        named.FillAllowed(mask);
        Assert.True(mask.IsAllowed(call));
        Assert.False(mask.IsAllowed(other));
        Assert.True(mask.IsAllowed('<'));
        Assert.True(named.Accept(call));
        Assert.False(named.IsComplete);
    }

    [Fact]
    public void JsonObjectConstraint_GreedyOutputParses()
    {
        // Picking the lowest allowed token at every step is the most adversarial "model" that still terminates for bounded grammars.
        var tok = new VocabTokenizer(Words);
        var g = Grammar.FromJsonSchema("""{"type":"object","properties":{"a":{"type":"integer"},"b":{"enum":["x","y"]},"c":{"type":"array","items":{"type":"boolean"},"maxItems":2}},"required":["a","b","c"]}""");
        var matcher = new GrammarConstraint(g, tok).CreateMatcher();
        var mask = new TokenMask(tok.VocabSize);
        var output = new List<byte>();
        for (int step = 0; step < 500 && !matcher.IsComplete; step++)
        {
            matcher.FillAllowed(mask);
            int id = Enumerable.Range(0, tok.VocabSize).Reverse().First(mask.IsAllowed);
            Assert.True(matcher.Accept(id));
            output.AddRange(tok.TokenBytes(id));
        }

        Assert.True(matcher.IsComplete);
        using var doc = JsonDocument.Parse(output.ToArray());
        Assert.Equal(JsonValueKind.Number, doc.RootElement.GetProperty("a").ValueKind);
    }
}
