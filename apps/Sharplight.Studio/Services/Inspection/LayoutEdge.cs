namespace Sharplight.Studio.Services.Inspection;

/// <summary>A dataflow edge of the graph view.</summary>
/// <param name="From">Index of the producing box in <see cref="GraphLayout.Nodes"/>.</param>
/// <param name="To">Index of the consuming box.</param>
/// <param name="Port">Where the edge enters the consumer, as a fraction of its width (inputs are spread across the top edge).</param>
internal readonly record struct LayoutEdge(int From, int To, double Port);
