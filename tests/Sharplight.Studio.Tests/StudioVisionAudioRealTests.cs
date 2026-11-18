using System.Globalization;
using System.Text.RegularExpressions;
using Sharplight.Audio;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.ViewModels.Audio;
using Sharplight.Studio.ViewModels.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The Vision and Audio pages end to end on real small models (skipped when a model is missing). The vision models look at the COCO
/// picture of two cats and two remote controls on a couch (the sample next to each model), the recognizer at a line of printed text, the
/// audio models at <c>jfk.wav</c>. The results must make sense for the picture and the recording, not only exist.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed partial class StudioVisionAudioRealTests
{
    [GeneratedRegex(@"^(\d+),(\d+) to (\d+),(\d+)$")]
    private static partial Regex Box();

    private static (string Folder, string Sample) Model(string name)
    {
        string folder = RealStudioSession.Require("vision/" + name);
        string sample = Path.Combine(folder, "sharplight-sample.png");
        Assert.SkipUnless(File.Exists(sample), $"{name}: no sample picture");
        return (folder, sample);
    }

    private static async Task<(VisionPageViewModel Page, RealStudioSession Session, string Sample)> OpenAsync(string name, string kind, VisionMode mode)
    {
        var (folder, sample) = Model(name);
        var session = new RealStudioSession();
        try
        {
            await session.LoadAsync(name, folder, kind, TestContext.Current.CancellationToken);
            var page = new VisionPageViewModel(session.Services, new Fakes.FakeNavigator());
            await page.ActivateAsync();
            Assert.Equal(mode, page.Mode);
            Assert.True(await page.Source.SetFileAsync(sample));
            return (page, session, sample);
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    private static async Task RunAsync(VisionPageViewModel page)
    {
        page.RunCommand.Execute(null);
        await RealStudioSession.FinishAsync(page);
        await page.Refresh.WaitAsync(TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task Classification_RanksCatLabelsFirst_ForThePictureOfTwoCats()
    {
        var (page, session, _) = await OpenAsync("vit-base-patch16-224", ModelKinds.ImageClassification, VisionMode.Classify);
        using (session)
        {
            await RunAsync(page);
            Assert.Equal(5, page.Rows.Count);
            Assert.Equal(page.Rows.OrderByDescending(row => row.Score), page.Rows);
            Assert.Contains("cat", page.Rows[0].Label, StringComparison.OrdinalIgnoreCase);
            Assert.True(page.Rows.Count(row => row.Label.Contains("cat", StringComparison.OrdinalIgnoreCase)) >= 2, string.Join(", ", page.Rows.Select(row => row.Label)));
            Assert.InRange(page.Rows[0].Score, 0.2f, 1f);
        }
    }

    [Theory]
    [InlineData("detr-resnet-50")]
    [InlineData("rtdetr_r18vd")]
    public async Task Detection_FindsBothCatsAndTheRemotes_AndDrawsTheBoxes(string name)
    {
        var (page, session, sample) = await OpenAsync(name, ModelKinds.ObjectDetection, VisionMode.Detect);
        using (session)
        {
            await RunAsync(page);
            var cats = page.Rows.Where(row => row.Label == "cat").ToList();
            Assert.True(cats.Count == 2, string.Join(", ", page.Rows.Select(row => $"{row.Label} {row.ScoreText} {row.Detail}")));
            Assert.Contains(page.Rows, row => row.Label == "remote");
            var centers = cats.Select(row =>
            {
                var match = Box().Match(row.Detail);
                Assert.True(match.Success, row.Detail);
                int[] v = [.. match.Groups.Cast<Group>().Skip(1).Select(group => int.Parse(group.Value, CultureInfo.InvariantCulture))];
                Assert.True(v[2] > v[0] + 100 && v[3] > v[1] + 100, $"a cat-sized box: {row.Detail}");
                Assert.True(v[0] >= -2 && v[1] >= -2 && v[2] <= 642 && v[3] <= 482, row.Detail);
                return (v[0] + v[2]) / 2;
            }).Order().ToList();
            Assert.True(centers[0] < 320 && centers[1] > 320, $"one cat left and one right: {string.Join(", ", centers)}");

            // The overlay is the picture with the boxes painted over it.
            var original = PngCodec.Decode(await File.ReadAllBytesAsync(sample, TestContext.Current.CancellationToken));
            var annotated = PngCodec.Decode(page.MainPng!);
            Assert.Equal((original.Width, original.Height), (annotated.Width, annotated.Height));
            Assert.NotEqual(original.Rgb, annotated.Rgb);

            // Raising the threshold hides objects without another run.
            int before = page.Rows.Count;
            page.Panel.Apply(new Dictionary<string, string> { ["threshold"] = "1" });
            await RealStudioSession.WaitFor(() => page.Rows.Count < before, 20);
        }
    }

    [Fact]
    public async Task Depth_GivesAColoredMapWithRealRange_OfTheSamePictureSize()
    {
        var (page, session, _) = await OpenAsync("depth-anything-v2-small", ModelKinds.DepthEstimation, VisionMode.Depth);
        using (session)
        {
            await RunAsync(page);
            Assert.NotNull(page.DepthPng);
            var depth = PngCodec.Decode(page.DepthPng!);
            Assert.Equal((640, 480), (depth.Width, depth.Height));
            byte[] pixels = depth.Rgb;
            Assert.True(pixels.Max() - pixels.Min() > 100, "the depth map uses a wide range of values");
            Assert.True(pixels.Distinct().Count() > 30);

            // Another color map redraws the map without another run.
            byte[] first = page.DepthPng!;
            page.Panel.Apply(new Dictionary<string, string> { ["invert"] = "true" });
            await RealStudioSession.WaitFor(() => !ReferenceEquals(page.DepthPng, first) && !first.AsSpan().SequenceEqual(page.DepthPng), 20);
        }
    }

    [Fact]
    public async Task Segmentation_GivesMasksOfObjectSize_AndPaintsThemOverThePicture()
    {
        var (page, session, sample) = await OpenAsync("sam2.1-hiera-tiny", ModelKinds.ImageSegmentation, VisionMode.Segment);
        using (session)
        {
            await RunAsync(page);
            Assert.NotEmpty(page.Rows);
            var best = page.Rows.OrderByDescending(row => row.Score).First();
            Assert.True(best.Score > 0.3);
            long area = long.Parse(best.Detail.Split(' ')[0], CultureInfo.InvariantCulture);
            Assert.InRange(area, 640 * 480 / 100, 640 * 480 * 95 / 100);
            var original = PngCodec.Decode(await File.ReadAllBytesAsync(sample, TestContext.Current.CancellationToken));
            Assert.NotEqual(original.Rgb, PngCodec.Decode(page.MainPng!).Rgb);
        }
    }

    [Fact]
    public async Task Ocr_ReadsThePrintedLine()
    {
        var (page, session, _) = await OpenAsync("trocr-base-printed", ModelKinds.Ocr, VisionMode.Ocr);
        using (session)
        {
            await RunAsync(page);
            Assert.Contains("sharplight", page.ResultText, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("printed text", page.ResultText, StringComparison.OrdinalIgnoreCase);
            Assert.Single(page.Gallery.Items);
        }
    }

    [Theory]
    [InlineData("audio-models/encodec_24khz", "encodec")]
    [InlineData("audio-models/mimi", "mimi")]
    public async Task Codec_RoundTripsJfk_AndMoreCodebooksReconstructBetter(string relative, string id)
    {
        string codec = RealStudioSession.Require(relative);
        string jfk = RealStudioSession.Require("audio/jfk.wav");
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        await session.LoadAsync(id, codec, ModelKinds.AudioCodec, ct);

        var page = new AudioPageViewModel(session.Services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal(AudioMode.Codec, page.Mode);
        Assert.True(await page.SetAudioAsync(jfk));
        double[] errors = new double[2];
        string[] codebooks = ["2", "8"];
        for (int i = 0; i < 2; i++)
        {
            page.Panel.Apply(new Dictionary<string, string> { ["codebooks"] = codebooks[i] });
            page.RunCommand.Execute(null);
            await RealStudioSession.FinishAsync(page);
            await RealStudioSession.WaitFor(() => page.HasResult);
            var match = Regex.Match(page.ResultSummary, @"error: (-?[\d.]+) dB");
            Assert.True(match.Success, page.ResultSummary);
            errors[i] = double.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
            Assert.Contains($"{codebooks[i]} codebooks", page.ResultSummary, StringComparison.Ordinal);
        }

        Assert.True(errors[1] < errors[0], $"2 codebooks {errors[0]} dB, 8 codebooks {errors[1]} dB");
        var output = page.Gallery.Items[0];
        var rebuilt = WavFile.Read(output.Files.Single(file => file.Name == "reconstruction.wav").Path);
        double seconds = WavFile.Read(jfk).Duration.TotalSeconds;
        Assert.Equal(seconds, rebuilt.Duration.TotalSeconds, 0.5);
        Assert.True(RealStudioSession.IsAudible(rebuilt.Samples.ToArray()));
        Assert.True(page.HasReconstruction);
    }

    [Fact]
    public async Task VoiceActivity_FindsTheSpeechOfJfk_InsideTheRecording()
    {
        string vad = RealStudioSession.Require("audio-models/silero-vad/silero_vad_16k_op15.onnx");
        string jfk = RealStudioSession.Require("audio/jfk.wav");
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        await session.LoadAsync("silero", vad, ModelKinds.VoiceActivity, ct);

        var page = new AudioPageViewModel(session.Services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal(AudioMode.VoiceActivity, page.Mode);
        Assert.True(await page.SetAudioAsync(jfk));
        page.RunCommand.Execute(null);
        await RealStudioSession.FinishAsync(page);
        await RealStudioSession.WaitFor(() => page.Segments.Count > 0);

        double seconds = WavFile.Read(jfk).Duration.TotalSeconds;
        Assert.InRange(page.Segments.Count, 1, 6);
        double speech = page.Segments.Sum(segment => (segment.End - segment.Start).TotalSeconds);
        Assert.InRange(speech, seconds * 0.4, seconds);
        Assert.All(page.Segments, segment => Assert.True(segment.Start >= TimeSpan.Zero && segment.End <= TimeSpan.FromSeconds(seconds + 0.1) && segment.End > segment.Start));
        Assert.Contains("segments:", page.ResultSummary, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Music_GeneratesAShortClipWithMusicGenSmall()
    {
        string musicgen = RealStudioSession.Require("audio-models/musicgen-small");
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        await session.LoadAsync("musicgen", musicgen, ModelKinds.MusicGeneration, ct);

        var page = new AudioPageViewModel(session.Services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal(AudioMode.Music, page.Mode);
        page.Prompt = "a calm piano melody";
        page.Panel.Apply(new Dictionary<string, string> { ["duration"] = "2", ["seed"] = "3" });
        page.RunCommand.Execute(null);
        await RealStudioSession.FinishAsync(page, 10);
        var clip = WavFile.Read(page.Gallery.Items[0].Files.Single(file => file.Name == "music.wav").Path);
        Assert.InRange(clip.Duration.TotalSeconds, 1.5, 3);
        Assert.True(RealStudioSession.IsAudible(clip.Samples.ToArray()));
    }
}
