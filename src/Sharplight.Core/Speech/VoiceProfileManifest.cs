namespace Sharplight.Speech;

/// <summary>The JSON side of a stored voice (<c>voice.json</c>): the descriptive fields and where each tensor lies in <c>tensors.bin</c>.</summary>
internal sealed class VoiceProfileManifest
{
    public int Version { get; set; } = 1;

    public string Name { get; set; } = "";

    public string Architecture { get; set; } = "";

    public string? ReferenceText { get; set; }

    public DateTimeOffset Created { get; set; }

    public int SampleRate { get; set; }

    public Dictionary<string, VoiceTensorEntry> Tensors { get; set; } = [];
}
