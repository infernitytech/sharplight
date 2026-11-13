using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>Parsing, normal form, evaluation and bounds of symbolic dimension expressions.</summary>
public class SymbolicExpressionTests
{
    private static SymbolicExpression P(string s) => SymbolicExpression.Parse(s);

    [Theory]
    [InlineData("N*12", "12*N")]
    [InlineData("N + 1 + N", "2*N+1")]
    [InlineData("(N+1)*(N-1)", "N*N-1")]
    [InlineData("floor(12*N/12)", "N")]
    [InlineData("floor((4*N+3)/2)", "2*N+1")]
    [InlineData("ceil((N+31)/32)", "ceil((N+31)/32)")]
    [InlineData("floor((N+31)/32)", "floor((N+31)/32)")]
    [InlineData("min(N+1, N)", "N")]
    [InlineData("max(T, N)", "max(N,T)")]
    [InlineData("min(200,nms#3)", "min(200,nms#3)")]
    [InlineData("3-3", "0")]
    [InlineData("T*N*2", "2*N*T")]
    public void Parse_Normalizes(string text, string canonical)
    {
        Assert.Equal(canonical, P(text).ToString());
        Assert.Equal(canonical, P(canonical).ToString());   // canonical text round-trips
    }

    [Fact]
    public void Evaluate_And_Bound()
    {
        var b = new Dictionary<string, long> { ["N"] = 5, ["T"] = 3 };
        Assert.Equal(60, P("12*N").Evaluate(b));
        Assert.Equal(1, P("ceil(N/32)").Evaluate(b));
        Assert.Equal(0, P("floor(N/32)").Evaluate(b));
        Assert.Equal(3, P("min(N,T)").Evaluate(b));
        Assert.Equal(2, P("N-T").Evaluate(b));
        Assert.Equal(60, SymbolicExpression.Evaluate("12*N", b));
        Assert.False(SymbolicExpression.TryEvaluate("12*M", b, out _));

        var options = new CompileOptions { SymbolBounds = new Dictionary<string, long> { ["N"] = 8 } };
        Assert.Equal(96, options.Bound("12*N"));
        Assert.Equal(1, options.Bound("ceil(N/32)"));
        Assert.Equal(7, options.Bound("N-1"));
        Assert.Equal(16, options.Bound("N+8"));
        Assert.Equal(10, options.Bound("10-N"));   // terms with a negative coefficient are bounded with their symbols at 0
        Assert.True(options.HasBound("N*4+1"));
        Assert.False(options.HasBound("N*M"));
    }

    [Fact]
    public void Dim_Arithmetic_ProducesExpressionDims()
    {
        var n = Dim.Sym("N");
        Assert.Equal("12*N", (n * 12).Symbol);
        Assert.Equal(Dim.Sym("N"), (n * 12).FloorDiv(12));
        Assert.Equal(Dim.Of(6), Dim.Of(2) * 3);
        Assert.Equal("N+1", (n + 1).Symbol);
        var shape = new Shape(n * 12, 3);
        Assert.Equal([24L, 3L], shape.Resolve(new Dictionary<string, long> { ["N"] = 2 }));
        Assert.Equal(72, shape.Elements(new Dictionary<string, long> { ["N"] = 2 }));
        Assert.Throws<FormatException>(() => Dim.Sym("N+"));
    }
}
