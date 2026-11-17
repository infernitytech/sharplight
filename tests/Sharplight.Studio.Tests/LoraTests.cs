using System.Text;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Gallery;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>LoRA adapters in Studio: library recognition, the load panel's slots, the Image page's reload and the gallery sidecar.</summary>
public sealed class LoraTests
{
    /// <summary>A safetensors file with the given tensor names (header only is enough for recognition).</summary>
    private static string WriteSafetensors(string path, params string[] names)
    {
        var entries = names.Select((name, index) => $"\"{name}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{index * 4},{index * 4 + 4}]}}");
        byte[] header = Encoding.UTF8.GetBytes("{" + string.Join(",", entries) + "}");
        using var stream = File.Create(path);
        stream.Write(BitConverter.GetBytes((long)header.Length));
        stream.Write(header);
        stream.Write(new byte[names.Length * 4]);
        return path;
    }

    [Theory]
    [InlineData("lora_unet_down_blocks_0.lora_down.weight", "lora_unet_down_blocks_0.lora_up.weight")]
    [InlineData("unet.down_blocks.0.attn.to_q.lora_A.weight", "unet.down_blocks.0.attn.to_q.lora_B.weight")]
    public void Library_RecognizesKohyaAndPeftFilesAsLoras(string down, string up)
    {
        using var services = new TestServices();
        string lora = WriteSafetensors(services.Directory + "/style.safetensors", down, up);
        string checkpoint = WriteSafetensors(services.Directory + "/model.safetensors", "model.diffusion_model.input_blocks.0.0.weight");

        Assert.Equal(ModelKindCatalog.Lora, ImageModelProbe.KindOf(lora));
        Assert.Equal(ModelKindCatalog.Lora, ModelInspector.Inspect(lora).Kind);
        Assert.Equal(ModelKinds.ImageGeneration, ImageModelProbe.KindOf(checkpoint));
    }

    [Fact]
    public void LoadPanel_PassesTheSlotsAsLoraRoles_WithTheirStrength()
    {
        using var services = new TestServices();
        string known = WriteSafetensors(services.Directory + "/known.safetensors", "a.lora_down.weight", "a.lora_up.weight");
        string other = WriteSafetensors(services.Directory + "/other.safetensors", "a.lora_down.weight", "a.lora_up.weight");
        var checkpoint = new LibraryEntry { Id = "sd", DisplayName = "sd", Path = WriteSafetensors(services.Directory + "/sd.safetensors", "model.diffusion_model.x"), Kind = ModelKinds.ImageGeneration };
        var libraryLora = new LibraryEntry { Id = "known", DisplayName = "known", Path = known, Kind = ModelKindCatalog.Lora };
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(checkpoint), services.Services, _ => { }, () => { }, () => [checkpoint, libraryLora]);

        Assert.True(panel.SupportsLoras);
        Assert.Equal([libraryLora], panel.Loras.Available);
        panel.Loras.AddLibraryCommand.Execute(libraryLora);
        panel.Loras.Add(other, 0.5);
        panel.Loras.Add(other, 0.9);
        Assert.Empty(panel.Loras.Available);

        var auxiliary = panel.ToRequest().Auxiliary!;
        Assert.Equal([("lora@1", known), ("lora:2@0.5", other)], auxiliary);
        Assert.Equal(["known", "other"], LoraRoles.Adapters(auxiliary).Select(lora => lora.Name));
        Assert.Equal([1f, 0.5f], LoraRoles.Adapters(auxiliary).Select(lora => lora.Scale));

        panel.Loras.Slots[0].RemoveCommand.Execute(null);
        Assert.Equal([("lora@0.5", other)], panel.ToRequest().Auxiliary!);
    }

    [Fact]
    public async Task ImagePage_ReloadsWithTheEditedLoras_RecordsThemInTheSidecar_AndReusesThem()
    {
        using var services = new TestServices();
        string style = WriteSafetensors(services.Directory + "/style.safetensors", "a.lora_down.weight", "a.lora_up.weight");
        var model = new FakeModel("sd-tiny") { Kind = ModelKinds.ImageGeneration, Auxiliary = [("lora@0.8", style)] };
        model.Clients[typeof(IImageGenerator)] = new FakeImageGenerator();
        services.Runtime.Add(model);
        var page = new ImagePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();

        Assert.Equal(["style"], page.Loras.Slots.Select(slot => slot.Name));
        Assert.Equal(0.8, page.Loras.Slots[0].Scale, 3);
        Assert.False(page.LorasPending);

        // A run records the LoRAs in its sidecar.
        page.Prompt = "a lighthouse";
        page.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == "seed").Value = "3";
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        var stored = new ImageGalleryStore(services.Services.Pages.GenerationsDirectory("images")).Load().Single();
        Assert.Equal([new GalleryLora { Name = "style", Scale = 0.8f, Path = style }], stored.Loras);

        // Changing a strength makes the model reload with the new role.
        page.Loras.Slots[0].Scale = 0.4;
        Assert.True(page.LorasPending);
        Assert.True(page.ApplyLorasCommand.CanExecute(null));
        await page.ApplyLorasCommand.ExecuteAsync(null);
        Assert.Equal("lora@0.4", services.Runtime.Requests.Last().Auxiliary!.Single().Role);
        Assert.Equal(0.4, page.Loras.Slots[0].Scale, 3);
        Assert.False(page.LorasPending);

        // Reusing the stored image's settings brings its LoRAs back and asks for a reload.
        page.Loras.Slots[0].RemoveCommand.Execute(null);
        page.Generations.ReuseSettingsCommand.Execute(page.Generations.Visible.Single());
        Assert.Equal(0.8, Assert.Single(page.Loras.Slots).Scale, 3);
        Assert.True(page.LorasPending);
    }
}
