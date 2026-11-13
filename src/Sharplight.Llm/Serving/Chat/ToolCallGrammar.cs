using System.Text.Json;
using Sharplight.Llm.Structured;

namespace Sharplight.Llm.Serving.Chat;

/// <summary>
/// Grammar for tool calls in a model family's syntax (<see cref="IToolCallFormat"/>): the call object
/// <c>{"name": "&lt;tool&gt;", "arguments": {..}}</c> with the arguments constrained by the tool's parameter schema, wrapped
/// the way the format writes it. Only the built-in formats are known; others are not constrained.
/// </summary>
internal static class ToolCallGrammar
{
    /// <summary>Marker texts that may be single special tokens in a vocabulary and must stay producible under a grammar.</summary>
    public static IReadOnlyList<string> Markers { get; } = ["<tool_call>", "</tool_call>", "[TOOL_CALLS]", "<|python_tag|>", "<think>", "</think>"];

    /// <summary>One or more calls in <paramref name="format"/>, or null when the format is not a built-in one.</summary>
    /// <param name="builder">Receives the rules.</param>
    /// <param name="format">The model's tool-call syntax.</param>
    /// <param name="tools">The callable tools.</param>
    /// <param name="afterStartMarker">The start marker was already written (the grammar is triggered by it).</param>
    /// <exception cref="NotSupportedException">A parameter schema uses an unsupported construct.</exception>
    /// <exception cref="JsonException">A parameter schema is not JSON.</exception>
    public static GrammarExpr? Calls(GrammarBuilder builder, IToolCallFormat format, IReadOnlyList<ToolSpec> tools, bool afterStartMarker)
    {
        JsonSchemaCompiler.DefineCommonRules(builder);
        var ws = JsonSchemaCompiler.Whitespace();
        var lit = GrammarExpr.Literal;
        switch (format.Name)
        {
            case "hermes":
            {
                var call = Call(builder, tools, "arguments");
                var rest = GrammarExpr.Seq(ws, call, ws, lit("</tool_call>"));
                var block = GrammarExpr.Seq(lit("<tool_call>"), rest);
                return GrammarExpr.Seq(afterStartMarker ? rest : block, GrammarExpr.Star(GrammarExpr.Seq(ws, block)));
            }

            case "mistral":
            {
                var call = Call(builder, tools, "arguments");
                var list = GrammarExpr.Seq(lit("["), ws, call, GrammarExpr.Star(GrammarExpr.Seq(ws, lit(","), ws, call)), ws, lit("]"));
                return afterStartMarker ? GrammarExpr.Seq(ws, list) : GrammarExpr.Seq(lit("[TOOL_CALLS]"), ws, list);
            }

            case "llama3":
            {
                var call = Call(builder, tools, "parameters");
                return afterStartMarker ? GrammarExpr.Seq(ws, call) : GrammarExpr.Seq(GrammarExpr.Optional(lit("<|python_tag|>")), call);
            }

            default:
                return null;
        }
    }

    /// <summary>The call object of any of <paramref name="tools"/>; the name decides the arguments schema.</summary>
    private static GrammarExpr Call(GrammarBuilder builder, IReadOnlyList<ToolSpec> tools, string argumentsKey)
    {
        var ws = JsonSchemaCompiler.Whitespace();
        var lit = GrammarExpr.Literal;
        var alternatives = new List<GrammarExpr>();
        foreach (var tool in tools)
        {
            GrammarExpr arguments;
            if (string.IsNullOrWhiteSpace(tool.ParametersJson))
            {
                arguments = JsonSchemaCompiler.AnyObject();
            }
            else
            {
                using var doc = JsonDocument.Parse(tool.ParametersJson);
                arguments = new JsonSchemaCompiler(builder, doc.RootElement).Compile(doc.RootElement);
            }

            alternatives.Add(GrammarExpr.Seq(
                lit("{"), ws, lit("\"name\""), ws, lit(":"), ws, lit(ToolCallFormats.JsonString(tool.Name)), ws, lit(","), ws,
                lit($"\"{argumentsKey}\""), ws, lit(":"), ws, arguments, ws, lit("}")));
        }

        return GrammarExpr.Choice(alternatives);
    }
}
