using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Video;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A video decoder that "reads" any <c>.mp4</c> path as the given frames (no FFmpeg needed).</summary>
internal sealed class FakeVideoDecoder(IReadOnlyList<ImageInput> frames, double framesPerSecond) : IVideoDecoder
{
    public bool IsAvailable => true;

    public IReadOnlyList<string> FileExtensions { get; } = [".mp4"];

    public Task<VideoInfo> ProbeAsync(string path, CancellationToken cancellationToken = default) => Task.FromResult(new VideoInfo
    {
        Container = "fake",
        VideoCodec = "fake",
        Width = frames[0].Width,
        Height = frames[0].Height,
        FramesPerSecond = framesPerSecond,
        Duration = TimeSpan.FromSeconds(frames.Count / framesPerSecond),
        FrameCount = frames.Count,
    });

    public async IAsyncEnumerable<VideoFrame> ReadFramesAsync(string path, VideoDecodeOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        int count = Math.Min(frames.Count, options?.MaxFrames ?? int.MaxValue);
        for (int i = 0; i < count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var frame = frames[i];
            yield return new VideoFrame
            {
                Index = i,
                Timestamp = TimeSpan.FromSeconds(i / framesPerSecond),
                Width = frame.Width,
                Height = frame.Height,
                Format = VideoFrameFormat.Rgb24,
                Data = frame.Rgb,
            };
            await Task.Yield();
        }
    }

    public Task<AudioBuffer?> ReadAudioAsync(string path, int? sampleRate = null, CancellationToken cancellationToken = default)
        => Task.FromResult<AudioBuffer?>(null);
}
