namespace Sharplight.Import.PyTorch;

/// <summary>The stack marker pushed by the MARK opcode.</summary>
internal sealed class PickleMark
{
    public static PickleMark Instance { get; } = new();

    private PickleMark()
    {
    }
}
