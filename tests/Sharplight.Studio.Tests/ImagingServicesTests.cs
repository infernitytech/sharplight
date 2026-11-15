using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>The image helpers of the pages, the library's recognition of image models and the kinds that became runnable.</summary>
public sealed class ImagingServicesTests
{
    [Fact]
    public void PngMetadata_RoundTrips_AndTheEngineStillDecodesTheFile()
    {
        var image = FakeImageGenerator.Gradient(16, 12, 1);
        byte[] png = PngCodec.Encode(image);
        byte[] tagged = PngMetadata.Embed(png, [new("Prompt", "a lighthouse"), new("Note", "Größe: 3 – ünïcode ✓")]);
        Assert.True(tagged.Length > png.Length);
        var text = PngMetadata.Read(tagged);
        Assert.Equal("a lighthouse", text["Prompt"]);
        Assert.Equal("Größe: 3 – ünïcode ✓", text["Note"]);
        Assert.Equal(image.Rgb, PngCodec.Decode(tagged).Rgb);
        Assert.Empty(PngMetadata.Read(png));
        Assert.Throws<InvalidDataException>(() => PngMetadata.Embed([1, 2, 3], []));
    }

    [Fact]
    public void ImageFiles_ReadPng_AndRefuseWhatNobodyDecodes()
    {
        using var directory = new TempDir();
        string path = directory.File("a.png");
        PngCodec.Write(path, FakeImageGenerator.Gradient(5, 4, 1));
        var image = ImageFiles.Read(path);
        Assert.Equal((5, 4), (image.Width, image.Height));
        Assert.True(ImageFiles.IsImage("x.JPG"));
        Assert.False(ImageFiles.IsImage("x.wav"));
        Assert.Throws<InvalidDataException>(() => ImageFiles.Decode([0, 1, 2, 3]));
    }

    [Fact]
    public void Annotator_DrawsBoxesInTheClassColor_WithoutTouchingTheOriginal()
    {
        var image = new ImageInput(40, 30, new byte[40 * 30 * 3]);
        var annotator = new ImageAnnotator(image);
        annotator.DrawBox(new BoundingBox(5, 10, 30, 25), 0, "cat 93%");
        var drawn = annotator.ToImage();
        Assert.All(image.Rgb, value => Assert.Equal(0, value));
        int top = 3 * (10 * 40 + 6);
        Assert.Equal(((byte)230, (byte)25, (byte)75), (drawn.Rgb[top], drawn.Rgb[top + 1], drawn.Rgb[top + 2]));
        int inside = 3 * (17 * 40 + 17);
        Assert.Equal(0, drawn.Rgb[inside]);
        Assert.Equal(0xE6194Bu, ImageAnnotator.ColorOf(0));
        Assert.Equal(ImageAnnotator.ColorOf(0), ImageAnnotator.ColorOf(16));
    }

    [Fact]
    public void DepthColormap_ScalesFromNearestToFarthest()
    {
        var map = new DepthMap(4, 1, [2f, 4f, 6f, 10f]);
        var gray = DepthColormap.Render(map, DepthColormap.Gray);
        Assert.Equal(0, gray.Rgb[0]);
        Assert.Equal(255, gray.Rgb[9]);
        var inverted = DepthColormap.Render(map, DepthColormap.Gray, invert: true);
        Assert.Equal(255, inverted.Rgb[0]);
        foreach (string name in DepthColormap.Names.Skip(1))
        {
            var colored = DepthColormap.Render(map, name);
            Assert.NotEqual((colored.Rgb[0], colored.Rgb[1]), (colored.Rgb[9], colored.Rgb[10]));
        }
    }

    [Fact]
    public void Library_RecognizesDiffusersFolders_SingleFiles_AndVisionCheckpoints()
    {
        using var directory = new TempDir();

        string diffusers = Directory.CreateDirectory(directory.File("sd-folder")).FullName;
        File.WriteAllText(Path.Combine(diffusers, "model_index.json"), "{\"_class_name\":\"StableDiffusionPipeline\"}");
        Assert.Equal(ModelKinds.ImageGeneration, ModelInspector.Inspect(diffusers).Kind);
        Assert.True(ModelInspector.Inspect(diffusers).KindInferred);

        // A single file is recognized by the tensor names of its header.
        string checkpoint = directory.File("model.safetensors");
        WriteSafetensors(checkpoint, "model.diffusion_model.input_blocks.0.0.weight");
        Assert.Equal(ModelKinds.ImageGeneration, ModelInspector.Inspect(checkpoint).Kind);
        string llm = directory.File("llm.safetensors");
        WriteSafetensors(llm, "model.layers.0.self_attn.q_proj.weight");
        Assert.Equal(ModelKindCatalog.Other, ModelKindOf(llm));
        Assert.Equal(ModelKinds.ImageGeneration, ModelInspector.Inspect(directory.File("v1-5-pruned-emaonly.ckpt").Touch()).Kind);
        Assert.Equal(ModelKindCatalog.Other, ModelInspector.Inspect(directory.File("weights.ckpt").Touch()).Kind);

        Assert.Equal(ModelKinds.ImageClassification, VisionFolder(directory, "vit", "{\"model_type\":\"vit\",\"architectures\":[\"ViTForImageClassification\"]}"));
        Assert.Equal(ModelKinds.ObjectDetection, VisionFolder(directory, "detr", "{\"model_type\":\"detr\",\"architectures\":[\"DetrForObjectDetection\"]}"));
        Assert.Equal(ModelKinds.ObjectDetection, VisionFolder(directory, "rtdetr", "{\"model_type\":\"rt_detr\",\"architectures\":[\"RTDetrForObjectDetection\"]}"));
        Assert.Equal(ModelKinds.DepthEstimation, VisionFolder(directory, "depth", "{\"model_type\":\"depth_anything\",\"architectures\":[\"DepthAnythingForDepthEstimation\"]}"));
        Assert.Equal(ModelKinds.ImageEmbedding, VisionFolder(directory, "clip", "{\"model_type\":\"clip\",\"architectures\":[\"CLIPModel\"]}"));
        Assert.Equal(ModelKinds.ImageEmbedding, VisionFolder(directory, "siglip", "{\"model_type\":\"siglip\",\"architectures\":[\"SiglipModel\"]}"));
        Assert.Equal(ModelKinds.ImageEmbedding, VisionFolder(directory, "dino", "{\"model_type\":\"dinov2\",\"architectures\":[\"Dinov2Model\"]}"));
        Assert.Equal(ModelKinds.TextGeneration, VisionFolder(directory, "llama", "{\"model_type\":\"llama\",\"architectures\":[\"LlamaForCausalLM\"]}"));
        Assert.Equal(ModelKinds.SpeechRecognition, VisionFolder(directory, "whisper", "{\"model_type\":\"whisper\"}"));
    }

    private static string ModelKindOf(string path) => ModelInspector.Inspect(path).Kind;

    private static string VisionFolder(TempDir directory, string name, string config)
    {
        string folder = Directory.CreateDirectory(directory.File(name)).FullName;
        File.WriteAllText(Path.Combine(folder, "config.json"), config);
        return ModelKindOf(folder);
    }

    private static void WriteSafetensors(string path, string tensor)
    {
        byte[] header = System.Text.Encoding.UTF8.GetBytes($"{{\"{tensor}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}}}");
        using var stream = File.Create(path);
        stream.Write(BitConverter.GetBytes((long)header.Length));
        stream.Write(header);
        stream.Write(new byte[4]);
    }

    [Fact]
    public void TheImageKinds_AreRunnable_AndHaveTaskClients()
    {
        foreach (string kind in new[] { ModelKinds.ImageGeneration, ModelKinds.ImageClassification, ModelKinds.ImageEmbedding, ModelKinds.ObjectDetection, ModelKinds.DepthEstimation })
        {
            Assert.True(ModelKindCatalog.For(kind).Runnable, kind);
            Assert.True(TaskKindClients.Handles(kind), kind);
        }

        Assert.False(ModelKindCatalog.For(ModelKinds.ImageSegmentation).Runnable);
        Assert.False(TaskKindClients.Handles(ModelKinds.ImageSegmentation));
    }
}

internal static class TempDirExtensions
{
    public static string Touch(this string path)
    {
        File.WriteAllBytes(path, [0]);
        return path;
    }
}
