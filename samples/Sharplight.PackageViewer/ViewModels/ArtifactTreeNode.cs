namespace Sharplight.PackageViewer.ViewModels;

/// <summary>A node of the artifact tree: a provider, a tier of a provider, or a set (leaf, <see cref="Set"/> not null).</summary>
/// <param name="Label">Main text.</param>
/// <param name="Detail">Secondary text (totals, distinguishing descriptor values).</param>
/// <param name="Children">Child nodes; empty for sets.</param>
/// <param name="Set">The set of a leaf.</param>
public sealed record ArtifactTreeNode(string Label, string Detail, IReadOnlyList<ArtifactTreeNode> Children, ArtifactSetViewModel? Set);
