namespace Sharplight.Speech;

/// <summary>Where one tensor of a stored voice lies in <c>tensors.bin</c>.</summary>
internal sealed class VoiceTensorEntry
{
    public string DataType { get; set; } = "";

    public int[] Shape { get; set; } = [];

    public long Offset { get; set; }

    public long Length { get; set; }
}
