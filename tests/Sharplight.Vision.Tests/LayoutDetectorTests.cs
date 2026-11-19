using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Tasks;

namespace Sharplight.Vision.Tests;

/// <summary>
/// PP-DocLayoutV3 (<c>PaddlePaddle/PP-DocLayoutV3_safetensors</c>) behind <see cref="IObjectDetector"/>, against transformers (<c>tools/layout-reference.py</c> writes the
/// synthetic page <c>sharplight-sample.png</c> and <c>sharplight-reference.json</c> next to the checkpoint): the same detections in the same reading order, boxes within a
/// pixel, masks of the same size, on CPU and Vulkan, also from an <c>.slm</c>. Skipped when the checkpoint or its reference is not present.
/// </summary>
public class LayoutDetectorTests
{
    private static readonly string Dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/vision-r10/pp-doclayoutv3");

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public async Task Layout_matches_transformers(bool gpu, bool package)
    {
        string refPath = Path.Combine(Dir, "sharplight-reference.json");
        Assert.SkipUnless(File.Exists(refPath), "PP-DocLayoutV3 or its reference is not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        var expected = JsonDocument.Parse(File.ReadAllText(refPath)).RootElement.GetProperty("detections").EnumerateArray().ToArray();
        var image = PngCodec.Read(Path.Combine(Dir, "sharplight-sample.png"));
        string scratch = Path.Combine(Path.GetTempPath(), "sl-slm-" + Guid.NewGuid().ToString("N"));
        try
        {
            string path = Dir;
            if (package)
            {
                Directory.CreateDirectory(scratch);
                path = Path.Combine(scratch, "model.slm");
                TestHost.Converter.Convert(Dir, path);
            }

            using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [slot!] });
            Assert.Equal("object-detection", model.Description.Kind);
            using var detector = new TaskClientProvider([.. Env.TaskFactories()]).Create<IObjectDetector>(model);
            var result = await detector.DetectAsync(image, new ObjectDetectionOptions { ScoreThreshold = 0.5f }, TestContext.Current.CancellationToken);
            TestContext.Current.SendDiagnosticMessage($"{(gpu ? "Vulkan" : "CPU")}{(package ? " (.slm)" : "")}: {string.Join("; ", result.Objects.Select(o => $"{o.ReadingOrder} {o.Label} {o.Score:f3}"))}");
            Assert.Equal(expected.Length, result.Objects.Count);
            double maxBox = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                var o = result.Objects[i];
                var box = expected[i].GetProperty("box").EnumerateArray().Select(v => v.GetSingle()).ToArray();
                Assert.Equal(expected[i].GetProperty("label").GetString(), o.Label);
                Assert.Equal(i, o.ReadingOrder);
                double diff = new[] { Math.Abs(o.Box.X1 - box[0]), Math.Abs(o.Box.Y1 - box[1]), Math.Abs(o.Box.X2 - box[2]), Math.Abs(o.Box.Y2 - box[3]) }.Max();
                maxBox = Math.Max(maxBox, diff);
                Assert.True(diff < 1.0, $"{o.Label}: box {o.Box} vs [{string.Join(", ", box)}]");
                Assert.True(Math.Abs(o.Score - expected[i].GetProperty("score").GetSingle()) < 0.02, $"{o.Label}: score {o.Score}");
                Assert.NotNull(o.Mask);
                Assert.True(o.Mask.Area > 0.3 * o.Box.Area, $"{o.Label}: mask area {o.Mask.Area} of box {o.Box.Area}");
            }

            TestContext.Current.SendDiagnosticMessage($"max box difference {maxBox:f3} px");
        }
        finally
        {
            if (Directory.Exists(scratch))
            {
                Directory.Delete(scratch, recursive: true);
            }
        }
    }
}
