namespace Sharplight.Hub;

/// <summary>One GGUF model of a repository: its tag and its files (all shards of a split model, in order).</summary>
internal sealed record GgufVariant(string Label, IReadOnlyList<string> Paths);
