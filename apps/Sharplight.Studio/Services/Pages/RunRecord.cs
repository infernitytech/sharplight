namespace Sharplight.Studio.Services.Pages;

/// <summary>One run of an inference page, as the history and the outputs gallery keep it.</summary>
internal sealed record RunRecord
{
    /// <summary>A unique id, also the name of the run's output directory.</summary>
    public required string Id { get; init; }

    /// <summary>What the run was for (the input's name or the start of the text).</summary>
    public required string Title { get; init; }

    /// <summary>When the run started.</summary>
    public DateTimeOffset Created { get; init; } = DateTimeOffset.Now;

    /// <summary>The id of the model that ran it.</summary>
    public string Model { get; init; } = "";

    /// <summary>How it ended.</summary>
    public RunOutcome Outcome { get; init; }

    /// <summary>How long it took, in seconds.</summary>
    public double Seconds { get; init; }

    /// <summary>The failure message, when it failed.</summary>
    public string? Error { get; init; }

    /// <summary>What the run was given (for example the input file), shown in the history.</summary>
    public Dictionary<string, string> Input { get; init; } = [];

    /// <summary>The parameter values of the run (a preset), restored with it.</summary>
    public Dictionary<string, string> Options { get; init; } = [];

    /// <summary>Facts about the output, shown on its gallery card (language, duration, speed, ...).</summary>
    public Dictionary<string, string> Metadata { get; init; } = [];

    /// <summary>A short excerpt of the output.</summary>
    public string? Preview { get; init; }

    /// <summary>The output files, by name, in the run's output directory. The first is the primary one.</summary>
    public List<string> Files { get; init; } = [];
}
