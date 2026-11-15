namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tests that load full-size checkpoints (several GiB each) onto the shared Vulkan device. They run one at a time and apart
/// from the other test collections, so their models never have to fit on the device together.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class LargeModelCollection
{
    /// <summary>The collection name.</summary>
    public const string Name = "Large models";
}
