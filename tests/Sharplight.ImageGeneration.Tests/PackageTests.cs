using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Vulkan;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>Image-generation models in <c>.slm</c> packages: conversion, description, loading, other sizes, precompilation.</summary>
public sealed class PackageTests
{
    private static string TempPackage(string name)
    {
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "imagegen-tests");
        Directory.CreateDirectory(folder);
        return Path.Combine(folder, $"{name}-{Guid.NewGuid():N}.slm");
    }

    [Theory]
    [InlineData("sd1")]
    [InlineData("sdxl")]
    public async Task ConvertedPackageGeneratesTheSameImages(string family)
    {
        string diffusers = Env.TinyDiffusersOf(family);
        Assert.SkipUnless(Directory.Exists(diffusers), $"tiny {family} pipeline missing");
        _ = Env.Services;
        string path = TempPackage("tiny-" + family);
        try
        {
            ModelConverter.Convert(diffusers, path);
            var info = ModelPackageReader.Open(path);
            Assert.True(info.Definition.HasDescription);
            Assert.Equal(ModelKinds.ImageGeneration, info.Definition.Description.Kind);
            Assert.Equal(["text-encoder", "tokenizer", "denoiser", "vae", "scheduler"], info.Definition.Description.Components.Select(c => c.Role));
            Assert.Contains(info.Attachments, a => a.Name == "scheduler/source-config.json");

            var request = new ImageGenerationRequest { Prompt = "a red cube", NegativePrompt = "blurry" };
            var options = new ImageGenerationOptions { Steps = 3, Seed = 5, Scheduler = Samplers.DpmPlusPlus2MKarras };
            var ct = TestContext.Current.CancellationToken;
            byte[] expected;
            using (var folder = Env.Load(diffusers, Env.Cpu()))
            using (var generator = Env.Clients.Create<IImageGenerator>(folder))
            {
                expected = (await generator.GenerateAsync(request, options, ct)).Images[0].Rgb;
            }

            using var package = Env.Load(path, Env.Cpu());
            using var fromPackage = Env.Clients.Create<IImageGenerator>(package);
            Assert.Equal(expected, (await fromPackage.GenerateAsync(request, options, ct)).Images[0].Rgb);

            // Another size rebuilds the graphs from the package's canonical metadata and weights.
            var wide = await fromPackage.GenerateAsync(request, options with { Width = 128 }, ct);
            Assert.Equal(128, wide.Images[0].Width);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task PrecompiledPackageLoadsWithoutCompiling()
    {
        Assert.SkipUnless(Directory.Exists(Env.Tiny), "tiny-random-sd missing");
        var vulkan = new VulkanProvider();
        Assert.SkipUnless(vulkan.CheckAvailability().IsAvailable, "no Vulkan");
        var device = vulkan.GetDevices().FirstOrDefault(d => d.Kind == Runtime.DeviceKind.DiscreteGpu);
        Assert.SkipWhen(device is null, "no discrete GPU");
        _ = Env.Services;
        string path = TempPackage("tiny-sd-precompiled");
        try
        {
            ModelConverter.Convert(Env.TinyDiffusers, path);
            var result = await PackagePrecompiler.PrecompileAsync(path, vulkan, device!, cancellationToken: TestContext.Current.CancellationToken);
            Assert.NotEmpty(result.StoredSets);
            var info = ModelPackageReader.Open(path);
            Assert.NotEmpty(info.ArtifactSets);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
