namespace Sharplight.Studio.Models;

/// <summary>What a video generation run was given and produced (<c>parameters.json</c> of a run).</summary>
internal sealed class VideoParametersFile
{
    /// <summary>The model that generated the video.</summary>
    public string Model { get; set; } = "";

    /// <summary>The prompt.</summary>
    public string Prompt { get; set; } = "";

    /// <summary>The negative prompt, or empty.</summary>
    public string NegativePrompt { get; set; } = "";

    /// <summary>The seed of the run.</summary>
    public long Seed { get; set; }

    /// <summary>Frame width in pixels.</summary>
    public int Width { get; set; }

    /// <summary>Frame height in pixels.</summary>
    public int Height { get; set; }

    /// <summary>The number of frames.</summary>
    public int Frames { get; set; }

    /// <summary>Frames per second of the result.</summary>
    public double FramesPerSecond { get; set; }

    /// <summary>Whether a start image conditioned the run (image to video).</summary>
    public bool ImageToVideo { get; set; }

    /// <summary>The page's parameter values, as a preset stores them (restore the page's panel with them).</summary>
    public Dictionary<string, string> Options { get; set; } = [];
}
