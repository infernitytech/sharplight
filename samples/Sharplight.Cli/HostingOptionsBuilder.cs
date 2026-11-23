using System.Globalization;
using Sharplight.Hosting;
using Sharplight.Samples;
using Sharplight.Samples.Hub;

namespace Sharplight.Cli;

/// <summary>
/// Turns the <c>serve</c> command line, and the optional hosting JSON file of <c>--config</c>, into <see cref="ModelHostingOptions"/>.
/// The file's settings are the base; options given on the command line win (model defaults only when they differ from the command
/// line's built-in defaults, so a file's <c>defaults</c> object is not overwritten by unspecified options).
/// </summary>
internal static class HostingOptionsBuilder
{
    private const string DefaultReserve = "1G";

    /// <exception cref="SampleException">The configuration is unreadable, a model setting is invalid or model names collide.</exception>
    public static ModelHostingOptions Build(ServeSettings settings)
    {
        var options = settings.ConfigPath is { } path ? Load(path) : new ModelHostingOptions { Reserve = MemorySize.Parse(DefaultReserve) };
        bool fromFile = settings.ConfigPath is not null;
        ApplyDefaults(options.Defaults, settings, fromFile);

        options.Reserve = settings.VramReserveBytes ?? options.Reserve;
        options.Budget = settings.VramBudgetBytes ?? options.Budget;
        options.MaxLoadedModels = settings.MaxLoadedModels ?? options.MaxLoadedModels;
        options.Lazy |= settings.Lazy;
        options.IdleTimeoutSeconds = settings.IdleTimeoutSeconds ?? options.IdleTimeoutSeconds;
        options.OnDemand |= settings.OnDemand;
        foreach (var feature in settings.Runtime.DisabledFeatures.Where(feature => !options.DisabledFeatures.Contains(feature)))
        {
            options.DisabledFeatures.Add(feature);
        }

        var models = new List<HostedModelOptions>();
        foreach (var positional in settings.PositionalModels)
        {
            models.Add(new HostedModelOptions
            {
                Name = settings.PositionalModelName ?? DefaultName(positional),
                Path = positional,
                Workers = settings.Workers is { Count: > 0 } workers ? [.. workers] : null,
            });
        }

        models.AddRange(settings.ModelArguments.Select(ModelSpecParser.Parse));
        models.AddRange(options.Models);
        EnsureValid(models);
        options.Models = models;
        return options;
    }

    private static ModelHostingOptions Load(string path)
    {
        try
        {
            return ModelHostingJson.Load(path);
        }
        catch (Exception error) when (error is FileNotFoundException or FormatException)
        {
            throw new SampleException(error.Message, error);
        }
    }

    private static void ApplyDefaults(HostedModelDefaults defaults, ServeSettings settings, bool fromFile)
    {
        var runtime = settings.Runtime;
        // --provider names the device by its id (e.g. cuda:0); without it the index is a GPU of the default provider.
        string index = runtime.DeviceIndex.ToString(CultureInfo.InvariantCulture);
        var device = runtime.UseCpu ? DeviceSpecifications.Cpu : runtime.ProviderName is { } provider ? $"{provider}:{index}" : index;
        bool explicitDevice = runtime.UseCpu || runtime.DeviceIndex != 0 || runtime.ProviderName is not null;
        if (!fromFile || explicitDevice)
        {
            defaults.Device = device;
        }

        defaults.ContextLength = Pick(fromFile, runtime.ContextLength, RuntimeOptionSet.DefaultContextLength, defaults.ContextLength);
        defaults.BatchTokens = Pick(fromFile, runtime.BatchTokens, RuntimeOptionSet.DefaultBatchTokens, defaults.BatchTokens);
        defaults.Slots = Pick(fromFile, settings.Slots, ServeOptionSet.DefaultSlots, defaults.Slots);
        defaults.MaxQueue = Pick(fromFile, settings.MaxQueue, ServeOptionSet.DefaultMaxQueue, defaults.MaxQueue);
        defaults.PrefillChunk = Pick(fromFile, settings.PrefillChunk, ServeOptionSet.DefaultPrefillChunk, defaults.PrefillChunk);
        defaults.NgramDraftTokens = Pick(fromFile, settings.NgramDraftTokens, 0, defaults.NgramDraftTokens);
        defaults.MtpDraftTokens = Pick(fromFile, settings.MtpDraftTokens, 0, defaults.MtpDraftTokens);
        defaults.SpeculationMaxBatch = Pick(fromFile, settings.SpeculationMaxBatch, ServeOptionSet.DefaultSpeculationMaxBatch, defaults.SpeculationMaxBatch);
        defaults.KvCacheType = Pick(fromFile, runtime.KvCacheType.Name, RuntimeOptionSet.DefaultKvCacheType, defaults.KvCacheType ?? RuntimeOptionSet.DefaultKvCacheType);
        defaults.CpuOffload |= runtime.CpuOffload;
        defaults.ReducedPrecision |= runtime.ReducedPrecision;
        defaults.StepTokens = settings.StepTokens ?? defaults.StepTokens;
        defaults.MinStepTokens = settings.MinStepTokens ?? defaults.MinStepTokens;
        defaults.StepFloor = settings.StepFloor ?? defaults.StepFloor;
        defaults.StepRowsPerSequence = settings.StepRowsPerSequence ?? defaults.StepRowsPerSequence;
        defaults.KvCacheTokens = settings.KvCacheTokens ?? defaults.KvCacheTokens;
        defaults.PrefixCache &= !settings.NoPrefixCache;
        defaults.KvStateSnapshots = settings.KvStateSnapshots ?? defaults.KvStateSnapshots;
        defaults.Projector = settings.MmprojPath ?? defaults.Projector;
        defaults.ChatTemplate = settings.ChatTemplatePath ?? defaults.ChatTemplate;
    }

    /// <summary>The command-line value, unless a file is read and the value is the command line's own default.</summary>
    private static T Pick<T>(bool fromFile, T commandLine, T commandLineDefault, T file)
    {
        return fromFile && EqualityComparer<T>.Default.Equals(commandLine, commandLineDefault) ? file : commandLine;
    }

    /// <summary>A file's name without extension; a stored or Hub model keeps the name it was given by.</summary>
    private static string DefaultName(string model)
    {
        return ModelResolver.IsLocalPath(model) ? Path.GetFileNameWithoutExtension(model.TrimEnd('/', '\\')) : model;
    }

    private static void EnsureValid(List<HostedModelOptions> models)
    {
        foreach (var model in models.Where(model => string.IsNullOrWhiteSpace(model.Name) || string.IsNullOrWhiteSpace(model.Path)))
        {
            throw new SampleException($"Every model needs a name and a path (got name '{model.Name}', path '{model.Path}').");
        }

        var duplicate = models.GroupBy(model => model.Name).FirstOrDefault(group => group.Count() > 1);
        if (duplicate is not null)
        {
            throw new SampleException($"Model names must be unique; '{duplicate.Key}' is used more than once.");
        }
    }
}
