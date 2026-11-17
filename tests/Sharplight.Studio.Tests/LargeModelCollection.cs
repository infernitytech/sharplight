namespace Sharplight.Studio.Tests;

/// <summary>
/// The tests that load real checkpoints (several GiB each) onto the GPU through the engine runtime. They run one at a time and apart
/// from the other test collections, so their models never have to fit on the device together (on a 32 GB GPU the image models of
/// classes running in parallel did not fit).
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class LargeModelCollection
{
    /// <summary>The collection name.</summary>
    public const string Name = "Large models";
}
