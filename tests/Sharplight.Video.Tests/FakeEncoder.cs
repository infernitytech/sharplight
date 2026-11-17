namespace Sharplight.Video.Tests;

/// <summary>An encoder that only reports a format, a priority and availability.</summary>
internal sealed class FakeEncoder(string format, int priority, bool available) : IVideoEncoder
{
    public string Format => format;

    public IReadOnlyList<string> FileExtensions { get; } = ["." + format];

    public bool IsAvailable => available;

    public int Priority => priority;

    public Task EncodeAsync(VideoClip clip, Stream output, VideoEncodeOptions? options = null, CancellationToken cancellationToken = default) => Task.CompletedTask;
}
