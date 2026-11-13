namespace Sharplight.PackageViewer.ViewModels;

/// <summary>One bar of a graph's op histogram.</summary>
/// <param name="Op">Op name.</param>
/// <param name="Count">Nodes with this op.</param>
/// <param name="Percent">Share of all nodes of the graph, 0 to 100.</param>
public sealed record OpCountRow(string Op, int Count, double Percent);
