namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A clock that moves only when a test moves it.</summary>
internal sealed class FakeTime(DateTimeOffset start) : TimeProvider
{
    public DateTimeOffset Now { get; set; } = start;

    public override DateTimeOffset GetUtcNow() => Now;
}
