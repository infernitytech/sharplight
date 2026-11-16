using System.Net;

namespace Sharplight.Hub.Tests;

/// <summary>The mapping from Hub pipeline tags and libraries to model tasks, the task filter of a search and multi-component repositories.</summary>
public sealed class ModelTaskTests
{
    [Theory]
    [InlineData("text-generation", null, null, ModelTask.TextGeneration)]
    [InlineData("text2text-generation", "transformers", null, ModelTask.TextGeneration)]
    [InlineData("image-text-to-text", null, "org/Model-VL", ModelTask.VisionLanguage)]
    [InlineData("automatic-speech-recognition", null, null, ModelTask.SpeechToText)]
    [InlineData("text-to-speech", null, null, ModelTask.TextToSpeech)]
    [InlineData("text-to-audio", null, null, ModelTask.TextToSpeech)]
    [InlineData("text-to-image", "diffusers", null, ModelTask.ImageGeneration)]
    [InlineData("image-to-video", null, null, ModelTask.VideoGeneration)]
    [InlineData("image-classification", null, null, ModelTask.ImageClassification)]
    [InlineData("object-detection", null, null, ModelTask.ObjectDetection)]
    [InlineData("image-segmentation", null, null, ModelTask.Segmentation)]
    [InlineData("depth-estimation", null, null, ModelTask.DepthEstimation)]
    [InlineData("image-to-text", null, "microsoft/trocr-base", ModelTask.Ocr)]
    [InlineData("image-to-text", null, "org/caption-model", ModelTask.Ocr)]
    [InlineData("image-text-to-text", null, "stepfun/GOT-OCR-2.0", ModelTask.Ocr)]
    [InlineData("image-to-image", "diffusers", "org/Real-ESRGAN-upscaler", ModelTask.Upscaling)]
    [InlineData("image-to-image", "diffusers", "org/controlnet-canny", ModelTask.ImageGeneration)]
    [InlineData("audio-classification", null, null, ModelTask.Audio)]
    [InlineData("feature-extraction", "sentence-transformers", null, ModelTask.Embeddings)]
    [InlineData("sentence-similarity", null, null, ModelTask.Embeddings)]
    [InlineData("text-ranking", null, null, ModelTask.Embeddings)]
    [InlineData("time-series-forecasting", null, null, ModelTask.TimeSeries)]
    [InlineData("image-to-3d", null, null, ModelTask.ThreeD)]
    [InlineData("feature-extraction", null, "facebook/esm2_t6_8M_UR50D", ModelTask.Bio)]
    [InlineData("robotics", null, null, ModelTask.Robotics)]
    [InlineData(null, "diffusers", null, ModelTask.ImageGeneration)]
    [InlineData(null, "whisper.cpp", null, ModelTask.SpeechToText)]
    [InlineData(null, "gguf", null, ModelTask.Other)]
    [InlineData("tabular-classification", null, null, ModelTask.Other)]
    [InlineData(null, null, null, ModelTask.Other)]
    public void PipelineTagsAndLibrariesMapToTasks(string? pipeline, string? library, string? repository, ModelTask expected)
    {
        // The "image-to-text" caption model below has no OCR name: it stays in the group of its tag (OCR and captioning share a tag).
        Assert.Equal(expected, ModelTasks.From(pipeline, library, repository));
    }

    [Fact]
    public void EveryTagAppearsInOneTask()
    {
        var seen = new Dictionary<string, ModelTask>(StringComparer.OrdinalIgnoreCase);
        foreach (var task in ModelTasks.All)
        {
            foreach (var tag in ModelTasks.PipelineTags(task))
            {
                Assert.True(seen.TryAdd(tag, task), $"{tag} is listed for {seen.GetValueOrDefault(tag)} and {task}");
            }
        }

        Assert.Equal(ModelTasks.All.Count, Enum.GetValues<ModelTask>().Length);
        Assert.Empty(ModelTasks.PipelineTags(ModelTask.Other));
    }

    [Fact]
    public void SummaryExposesItsTask()
    {
        var summary = new HubModelSummary("a/whisper-small", 1, 1, false, null, "automatic-speech-recognition", null, [], []);
        Assert.Equal(ModelTask.SpeechToText, summary.Task);
    }

    private static string Listing(params (string Id, string Pipeline, long Downloads)[] models)
        => "[" + string.Join(",", models.Select(model =>
            $$"""{"id":"{{model.Id}}","downloads":{{model.Downloads}},"likes":1,"pipeline_tag":"{{model.Pipeline}}","tags":["license:apache-2.0"],"lastModified":"2026-10-01T00:00:00.000Z"}""")) + "]";

    [Fact]
    public async Task TaskSearchAsksOncePerPipelineTag_AndMergesTheAnswers()
    {
        using var hub = new FakeHub
        {
            Intercept = request => Uri.UnescapeDataString(request.RequestUri!.Query) switch
            {
                var query when query.Contains("pipeline_tag=text-to-speech", StringComparison.Ordinal)
                    => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(Listing(("a/tts-1", "text-to-speech", 50), ("a/tts-2", "text-to-speech", 5))) },
                var query when query.Contains("pipeline_tag=text-to-audio", StringComparison.Ordinal)
                    => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(Listing(("b/audio-1", "text-to-audio", 500), ("a/tts-1", "text-to-speech", 50))) },
                _ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("[]") },
            },
        };
        using var http = hub.CreateClient();
        var client = new HubClient(http, hub.Options());

        var results = await client.SearchAsync(new HubSearchQuery { Task = ModelTask.TextToSpeech, License = "apache-2.0", Limit = 10 }, TestContext.Current.CancellationToken);

        Assert.Equal(["b/audio-1", "a/tts-1", "a/tts-2"], results.Select(result => result.Id));
        var queries = hub.Requests.Select(request => Uri.UnescapeDataString(request.RequestUri!.Query)).ToList();
        Assert.Equal(2, queries.Count);
        Assert.All(queries, query => Assert.Contains("filter=license:apache-2.0", query, StringComparison.Ordinal));
        Assert.Equal("apache-2.0", results[0].License);
        Assert.NotNull(results[0].LastModified);
    }

    [Fact]
    public async Task TaskSearchKeepsOnlyRepositoriesOfTheTask()
    {
        using var hub = new FakeHub
        {
            Intercept = _ => new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(Listing(("x/Real-ESRGAN-x4", "image-to-image", 9), ("x/controlnet", "image-to-image", 99))),
            },
        };
        using var http = hub.CreateClient();
        var client = new HubClient(http, hub.Options());

        var upscalers = await client.SearchAsync(new HubSearchQuery { Task = ModelTask.Upscaling, Limit = 5 }, TestContext.Current.CancellationToken);

        Assert.Equal(["x/Real-ESRGAN-x4"], upscalers.Select(result => result.Id));
        var query = Uri.UnescapeDataString(hub.Requests[0].RequestUri!.Query);
        Assert.Contains("pipeline_tag=image-to-image", query, StringComparison.Ordinal);
        Assert.Contains("search=upscale", query, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(ModelFormat.Diffusers, "filter=diffusers")]
    [InlineData(ModelFormat.WhisperCpp, "filter=ggml")]
    [InlineData(ModelFormat.Onnx, "filter=onnx")]
    [InlineData(ModelFormat.Safetensors, "filter=safetensors")]
    public async Task FormatFilterMapsToHubTags(ModelFormat format, string expected)
    {
        using var hub = new FakeHub { Intercept = _ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("[]") } };
        using var http = hub.CreateClient();
        var client = new HubClient(http, hub.Options());

        await client.SearchAsync(new HubSearchQuery { Format = format, Sort = HubSortOrder.Modified }, TestContext.Current.CancellationToken);

        var query = Uri.UnescapeDataString(Assert.Single(hub.Requests).RequestUri!.Query);
        Assert.Contains(expected, query, StringComparison.Ordinal);
        Assert.Contains("sort=lastModified", query, StringComparison.Ordinal);
    }

    [Fact]
    public void ResultsReportMultiComponentFormats()
    {
        var results = HubJson.ParseSearch("""
            [
              {"id":"o/sd","siblings":[{"rfilename":"model_index.json"},{"rfilename":"unet/diffusion_pytorch_model.safetensors"}],"pipeline_tag":"text-to-image","library_name":"diffusers"},
              {"id":"o/whisper","siblings":[{"rfilename":"ggml-small.bin"},{"rfilename":"ggml-base.en-q5_1.bin"}]},
              {"id":"o/slm","siblings":[{"rfilename":"model.slm"}]}
            ]
            """);

        Assert.Contains(ModelFormat.Diffusers, results[0].Formats);
        Assert.Equal(ModelTask.ImageGeneration, results[0].Task);
        Assert.Equal([ModelFormat.WhisperCpp], results[1].Formats);
        Assert.Equal([ModelFormat.Slm], results[2].Formats);
    }

    [Fact]
    public void DiffusersPipelineIsOneModelWithItsComponents()
    {
        HubFile File(string path, long size) => new(path, size, null, null);
        var components = RepoLayouts.DiffusersComponents(
        [
            File("model_index.json", 10),
            File("unet/diffusion_pytorch_model.safetensors", 3_400),
            File("unet/diffusion_pytorch_model.bin", 3_400),
            File("unet/config.json", 5),
            File("vae/diffusion_pytorch_model.safetensors", 330),
            File("vae/diffusion_pytorch_model.fp16.safetensors", 165),
            File("text_encoder/model.safetensors", 490),
            File("scheduler/scheduler_config.json", 1),
        ]);

        Assert.Equal(["unet", "text_encoder", "vae"], components.Select(component => component.Name));
        // The safetensors copy counts once; a half-precision duplicate is left out.
        Assert.Equal(3_400, components[0].Bytes);
        Assert.Equal(330, components[2].Bytes);
        Assert.All(components, component => Assert.Equal(1, component.Files));
    }

    [Fact]
    public void WhisperCppFilesAreListedBySizeAndQuantization()
    {
        HubFile File(string path, long size) => new(path, size, null, null);
        var models = RepoLayouts.WhisperModels([File("ggml-small.bin", 466), File("ggml-base.en-q5_1.bin", 60), File("README.md", 1), File("models/ggml-x.bin", 9)]);

        Assert.Equal(["base.en-q5_1", "small"], models.Select(model => model.Name));
        Assert.True(RepoLayouts.IsWhisperCpp("ggml-tiny.bin"));
        Assert.False(RepoLayouts.IsWhisperCpp("models/ggml-tiny.bin"));
    }

    [Fact]
    public void MultiComponentRepositoriesArePulledAsOneModel()
    {
        Assert.Equal(ModelFormat.Diffusers, ModelFileSelector.DetectFormat(["model_index.json", "unet/diffusion_pytorch_model.safetensors"]));
        Assert.Equal(ModelFormat.WhisperCpp, ModelFileSelector.DetectFormat(["ggml-small.bin"]));
    }
}
