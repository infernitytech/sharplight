namespace Sharplight.Package.Import;

/// <summary>A graph without a language-model layout has a symbolic dimension that <see cref="PrecompileOptions.SymbolBounds"/> does not bound.</summary>
public sealed class MissingSymbolBoundException : InvalidOperationException
{
    /// <summary>Creates the exception for <paramref name="graph"/>, keeping the compiler's message.</summary>
    /// <param name="graph">Name of the graph that failed to compile.</param>
    /// <param name="inner">The compiler's error.</param>
    public MissingSymbolBoundException(string graph, Exception inner)
        : base($"Graph '{graph}': {inner.Message}", inner)
    {
        Graph = graph;
    }

    /// <summary>Name of the graph that failed to compile.</summary>
    public string Graph { get; }
}
