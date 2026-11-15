namespace Sharplight.Studio.Models;

/// <summary>
/// The result of a vision run as the page stores and exports it (<c>result.json</c> of a run): the mode, the image, and what the model
/// returned (class scores, boxes, depth range or an embedding).
/// </summary>
internal sealed class VisionResultFile
{
    /// <summary><c>classification</c>, <c>zero-shot</c>, <c>embedding</c>, <c>detection</c> or <c>depth</c>.</summary>
    public string Mode { get; set; } = "";

    /// <summary>The model that ran it.</summary>
    public string Model { get; set; } = "";

    /// <summary>The input image's file name.</summary>
    public string Image { get; set; } = "";

    /// <summary>The input image's width in pixels.</summary>
    public int Width { get; set; }

    /// <summary>The input image's height in pixels.</summary>
    public int Height { get; set; }

    /// <summary>The candidate labels of a zero-shot run.</summary>
    public List<string>? Candidates { get; set; }

    /// <summary>The classes, highest score first (classification and zero-shot).</summary>
    public List<VisionLabelRecord>? Labels { get; set; }

    /// <summary>The detected objects, best first. A stored run keeps every object the model found; an export has those above <see cref="Threshold"/>.</summary>
    public List<VisionBoxRecord>? Boxes { get; set; }

    /// <summary>The score threshold the objects were filtered with (detection).</summary>
    public float? Threshold { get; set; }

    /// <summary>The smallest depth value (depth estimation).</summary>
    public float? DepthMin { get; set; }

    /// <summary>The largest depth value (depth estimation).</summary>
    public float? DepthMax { get; set; }

    /// <summary>The length of the embedding vector (embedding).</summary>
    public int? Dimensions { get; set; }

    /// <summary>The embedding vector.</summary>
    public float[]? Embedding { get; set; }
}
