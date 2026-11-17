using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Llm.Serving.Speculative;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Hosting.Loaders;

/// <summary>Loads <see cref="ModelKind.Chat"/> models with <see cref="LlmChatModelLoader"/> (one engine and scheduler per model).</summary>
public sealed class ChatModelLoaderFactory : IHostedModelLoaderFactory
{
    /// <inheritdoc/>
    public bool CanLoad(HostedModelSettings model) => model.Kind == ModelKind.Chat;

    /// <inheritdoc/>
    public IHostedModelLoader CreateLoader(HostedModelLoadContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        var settings = context.Settings;
        var loader = context.Services.GetRequiredService<ModelLoader>();
        var auxiliary = settings.Projector is { } projector ? settings.Auxiliary.Values.Append(projector) : settings.Auxiliary.Values;
        return new PlacedModelLoader(ModelKind.Chat, context, () => FileFootprint.Estimate(settings.Path, auxiliary),
            placement => new LlmChatModelLoader(loader, settings.Name, settings.Path, CreateLoadOptions(context, placement), CreateChatOptions(context)));
    }

    private static LlmLoadOptions CreateLoadOptions(HostedModelLoadContext context, DevicePlacement placement)
    {
        var settings = context.Settings;
        var auxiliary = new Dictionary<string, string>(settings.Auxiliary, StringComparer.Ordinal);
        if (settings.Projector is { } projector)
        {
            auxiliary[AuxiliaryRoles.Mmproj] = projector;
        }

        var load = new LlmLoadOptions
        {
            Devices = [.. placement.Devices],
            Fallback = placement.Fallback,
            ContextLength = settings.ContextLength,
            MaxBatchTokens = settings.BatchTokens,
            KvCacheType = settings.KvCacheType is { } type ? DataType.Get(type) : null,
            AllowReducedPrecision = settings.ReducedPrecision,
            Auxiliary = auxiliary,
        };
        context.Options.ConfigureLoad?.Invoke(settings, load);
        return load;
    }

    private static ChatModelOptions CreateChatOptions(HostedModelLoadContext context)
    {
        var settings = context.Settings;
        var chat = new ChatModelOptions
        {
            Slots = settings.Slots,
            KvCacheTokens = settings.KvCacheTokens,
            Engine = new EngineOptions
            {
                MaxQueue = settings.MaxQueue,
                PrefillChunk = settings.PrefillChunk,
                MaxBatchTokens = settings.StepTokens ?? 0,
                SpeculationMaxBatch = settings.SpeculationMaxBatch,
            },
            Speculator = settings.NgramDraftTokens > 0 ? _ => new NgramSpeculator(settings.NgramDraftTokens) : null,
            MtpDraftTokens = settings.MtpDraftTokens,
            Service = new ModelServiceOptions { ChatTemplate = settings.ChatTemplate is { } path ? ReadTemplate(path) : null },
        };
        context.Options.ConfigureChat?.Invoke(settings, chat);
        return chat;
    }

    private static string ReadTemplate(string path)
    {
        return File.Exists(path) ? File.ReadAllText(path) : throw new FileNotFoundException($"Chat template '{path}' does not exist.", path);
    }
}
