using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Tests;

public sealed class StorageTests
{
    [Fact]
    public async Task Conversations_RoundTripWithBranchesAndAttachments()
    {
        using var services = new TestServices();
        var store = services.Conversations;
        var user = new ChatMessageRecord
        {
            Role = ChatRole.User,
            Content = "What is in the picture?",
            Attachments =
            [
                new AttachmentRecord { Kind = AttachmentKind.Image, Name = "cat.png", MediaType = "image/png", Data = [1, 2, 3] },
                new AttachmentRecord { Kind = AttachmentKind.Document, Name = "notes.md", Text = "# Notes", TokenCount = 3, Truncated = true },
            ],
        };
        var answer = new ChatMessageRecord
        {
            ParentId = user.Id,
            Role = ChatRole.Assistant,
            Content = "A cat.",
            Reasoning = "Looks like fur.",
            ModelId = "qwen3",
            Stats = new MessageStats { PromptTokens = 10, CompletionTokens = 3, DecodeMs = 5, DraftTokens = 4, AcceptedDraftTokens = 2, Finish = "length" },
        };
        var conversation = new Conversation
        {
            Title = "Pictures",
            ModelId = "qwen3",
            SystemPrompt = "Be brief.",
            Sampling = new SamplingSettings { Temperature = 0.2, Seed = 7 },
            EnableThinking = false,
            Messages = [user, answer],
            CurrentLeafId = answer.Id,
        };

        await store.SaveAsync(conversation);
        var loaded = await store.LoadAsync(conversation.Id);

        Assert.NotNull(loaded);
        Assert.Equal(("Pictures", "qwen3", "Be brief.", false, answer.Id), (loaded.Title, loaded.ModelId, loaded.SystemPrompt, loaded.EnableThinking, loaded.CurrentLeafId));
        Assert.Equal(conversation.Sampling, loaded.Sampling);
        Assert.Equal(2, loaded.Messages.Count);
        var loadedUser = loaded.Messages.Single(message => message.Id == user.Id);
        Assert.Equal([1, 2, 3], loadedUser.Attachments[0].Data);
        Assert.Equal(("notes.md", "# Notes", true), (loadedUser.Attachments[1].Name, loadedUser.Attachments[1].Text, loadedUser.Attachments[1].Truncated));
        var loadedAnswer = loaded.Messages.Single(message => message.Id == answer.Id);
        Assert.Equal(answer.Stats, loadedAnswer.Stats);
        Assert.Equal(("Looks like fur.", user.Id), (loadedAnswer.Reasoning, loadedAnswer.ParentId));

        var summary = Assert.Single(await store.ListAsync());
        Assert.Equal((conversation.Id, "Pictures", 2), (summary.Id, summary.Title, summary.MessageCount));

        await store.SaveAsync(loaded with { Title = "Renamed", Messages = [user], CurrentLeafId = user.Id });
        Assert.Single((await store.LoadAsync(conversation.Id))!.Messages);

        await store.DeleteAsync(conversation.Id);
        Assert.Null(await store.LoadAsync(conversation.Id));
        Assert.Empty(await store.ListAsync());
    }

    [Fact]
    public async Task Stats_AreQueriedByTimeAndModel_AndRaiseRecordAdded()
    {
        using var services = new TestServices();
        var store = services.Stats;
        var added = new List<RequestRecord>();
        store.RecordAdded += (_, record) => added.Add(record);
        var now = DateTimeOffset.UtcNow;
        await store.AddAsync(new RequestRecord { Model = "a", Timestamp = now.AddDays(-2), Stats = new MessageStats { CompletionTokens = 1 } });
        await store.AddAsync(new RequestRecord { Model = "a", Timestamp = now, Source = RequestSource.Api, Device = "GPU", Stats = new MessageStats { CompletionTokens = 2, Finish = "length" } });
        await store.AddAsync(new RequestRecord { Model = "b", Timestamp = now, Stats = new MessageStats { CompletionTokens = 3 } });

        Assert.Equal(3, added.Count);
        Assert.Equal(2, (await store.QueryAsync(now.AddHours(-1), null)).Count);
        var a = Assert.Single(await store.QueryAsync(now.AddHours(-1), "a"));
        Assert.Equal((RequestSource.Api, "GPU", 2, "length"), (a.Source, a.Device, a.Stats.CompletionTokens, a.Stats.Finish));
        Assert.Equal(3, (await store.QueryAsync(now.AddDays(-3), null)).Count);

        await store.ClearAsync();
        Assert.Empty(await store.QueryAsync(DateTimeOffset.MinValue, null));
    }

    [Fact]
    public void Settings_ArePersistedAndReloaded()
    {
        using var dir = new TempDir();
        var path = dir.File("config/settings.json");
        var settings = new JsonSettingsService(path, null);
        int changes = 0;
        settings.Changed += (_, _) => changes++;
        settings.Update(current => current with
        {
            Theme = AppTheme.Light,
            Accent = "Ocean",
            DefaultLoad = current.DefaultLoad with { AllowReducedPrecision = true, MtpDraftTokens = 2 },
            Residency = current.Residency with { LoadOnDemand = true, IdleUnloadMinutes = 15 },
            Downloads = [new PendingDownload("Qwen/Qwen3-0.6B-GGUF:Q8_0", 600, true)],
        });

        Assert.Equal(1, changes);
        var reloaded = new JsonSettingsService(path, "/from-cli").Current;
        Assert.Equal((AppTheme.Light, "Ocean"), (reloaded.Theme, reloaded.Accent));
        Assert.True(reloaded.DefaultLoad.AllowReducedPrecision);
        Assert.Equal(2, reloaded.DefaultLoad.MtpDraftTokens);
        Assert.Equal((true, 15), (reloaded.Residency.LoadOnDemand, reloaded.Residency.IdleUnloadMinutes));
        Assert.Equal(new PendingDownload("Qwen/Qwen3-0.6B-GGUF:Q8_0", 600, true), Assert.Single(reloaded.Downloads));
        Assert.Equal("/from-cli", new JsonSettingsService(path, "/from-cli").StoreRoot);
    }

    [Fact]
    public void Settings_FallBackToDefaults_WhenTheFileIsBroken()
    {
        using var dir = new TempDir();
        File.WriteAllText(dir.File("settings.json"), "{ not json");
        Assert.Equal(new AppSettings().Theme, JsonSettingsService.Load(dir.File("settings.json")).Theme);
    }
}
