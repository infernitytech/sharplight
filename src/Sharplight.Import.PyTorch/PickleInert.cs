namespace Sharplight.Import.PyTorch;

/// <summary>A placeholder for an object of a module the caller declared inert: it accepts calls and state and does nothing with them.</summary>
internal sealed class PickleInert
{
    public static PickleInert Instance { get; } = new();

    private PickleInert()
    {
    }
}
