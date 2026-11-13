using Sharplight.Samples;

namespace Sharplight.Serve;

/// <summary>Merges command-line settings and the optional JSON configuration into a <see cref="ServerPlan"/>; command-line values win.</summary>
internal static class ServerPlanBuilder
{
    private const string DefaultVramReserve = "1G";

    /// <exception cref="SampleException">The configuration is unreadable, a model setting is invalid or model names collide.</exception>
    public static ServerPlan Build(ServeSettings settings)
    {
        var config = settings.ConfigPath is { } path ? ServerConfigLoader.Load(path) : null;
        var models = CollectModels(settings, config);

        long reserve = settings.VramReserveBytes
            ?? ParseConfigSize(config?.VramReserve, "vramReserve")
            ?? ByteSize.Parse(DefaultVramReserve);
        long? budget = settings.VramBudgetBytes ?? ParseConfigSize(config?.VramBudget, "vramBudget");

        return new ServerPlan(
            models,
            reserve,
            budget,
            settings.MaxLoadedModels ?? config?.MaxLoaded ?? 0,
            settings.Lazy || config?.Lazy == true,
            settings.IdleTimeoutSeconds ?? config?.IdleTimeout);
    }

    private static List<ModelSpec> CollectModels(ServeSettings settings, ServerConfig? config)
    {
        var models = new List<ModelSpec>();
        if (settings.PositionalModel is { } positional)
        {
            var name = settings.PositionalModelName ?? Path.GetFileNameWithoutExtension(positional.TrimEnd('/', '\\'));
            models.Add(new ModelSpec { Name = name, Path = positional });
        }

        models.AddRange(settings.ModelArguments.Select(ModelSpecParser.Parse));
        if (config?.Models is { } configured)
        {
            models.AddRange(configured);
        }

        EnsureValid(models);
        return models;
    }

    private static void EnsureValid(List<ModelSpec> models)
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

    private static long? ParseConfigSize(string? text, string key)
    {
        if (text is null)
        {
            return null;
        }

        try
        {
            return ByteSize.Parse(text);
        }
        catch (FormatException error)
        {
            throw new SampleException($"Config value '{key}': {error.Message}", error);
        }
    }
}
