namespace Sharplight.Import.PyTorch;

/// <summary>Options for reading a PyTorch checkpoint.</summary>
public sealed class TorchCheckpointOptions
{
    /// <summary>The defaults: only the tensor state-dict globals are allowed, and a nested <c>state_dict</c> entry is unwrapped.</summary>
    public static TorchCheckpointOptions Default { get; } = new();

    /// <summary>
    /// Module-name prefixes (for example <c>pytorch_lightning.</c>) whose globals are accepted as inert placeholders: calling or
    /// building them records nothing and runs nothing. Training checkpoints keep callback and config objects beside the weights;
    /// this lets such files load without allowing any behavior. Empty by default.
    /// </summary>
    public IReadOnlyList<string> InertModulePrefixes { get; init; } = [];

    /// <summary>
    /// When the top-level dictionary has no tensors but holds a dictionary under this key (training checkpoints:
    /// <c>state_dict</c>), the tensors are read from it. Null disables unwrapping.
    /// </summary>
    public string? StateDictKey { get; init; } = "state_dict";

    /// <summary>Upper bound on the size of the pickle stream in bytes; larger streams are refused.</summary>
    public long MaxPickleBytes { get; init; } = 1L << 30;

    /// <summary>Upper bound on the number of stack and memo entries; guards against pickles built to exhaust memory.</summary>
    public int MaxObjects { get; init; } = 16_000_000;
}
