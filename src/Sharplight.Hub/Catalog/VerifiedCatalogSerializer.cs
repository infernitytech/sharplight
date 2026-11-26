using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;

namespace Sharplight.Hub.Catalog;

/// <summary>Reads and writes <see cref="VerifiedCatalog"/> JSON.</summary>
public static class VerifiedCatalogSerializer
{
    /// <summary>Parses a catalog.</summary>
    /// <param name="json">The catalog file's text.</param>
    /// <exception cref="InvalidDataException">The text is not a catalog, or its schema version is newer than <see cref="VerifiedCatalog.CurrentSchemaVersion"/>.</exception>
    public static VerifiedCatalog Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        VerifiedCatalog? catalog;
        try
        {
            catalog = JsonSerializer.Deserialize(json, CatalogJsonContext.Default.VerifiedCatalog);
        }
        catch (JsonException error)
        {
            throw new InvalidDataException($"The verified catalog is not valid JSON: {error.Message}", error);
        }

        if (catalog is null)
        {
            throw new InvalidDataException("The verified catalog is empty.");
        }

        if (catalog.SchemaVersion > VerifiedCatalog.CurrentSchemaVersion || catalog.SchemaVersion < 1)
        {
            throw new InvalidDataException($"The verified catalog has schema version {catalog.SchemaVersion}; this version reads {VerifiedCatalog.CurrentSchemaVersion}.");
        }

        return catalog;
    }

    /// <summary>Writes <paramref name="catalog"/> as indented JSON with a final line break.</summary>
    /// <param name="catalog">The catalog.</param>
    public static string Serialize(VerifiedCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        return JsonSerializer.Serialize(catalog, CatalogJsonContext.Default.VerifiedCatalog).ReplaceLineEndings("\n") + "\n";
    }

    /// <summary>
    /// The JSON schema of the catalog file (draft 2020-12, generated from the types; <c>catalog/verified-models.schema.json</c>), with the
    /// patterns of ids, commits and checksums. Editors use it for completion; <see cref="CatalogValidator"/> checks the rest.
    /// </summary>
    public static string ExportSchema()
    {
        var options = new JsonSchemaExporterOptions { TransformSchemaNode = AddConstraints, TreatNullObliviousAsNonNullable = true };
        var schema = CatalogJsonContext.Default.VerifiedCatalog.GetJsonSchemaAsNode(options).AsObject();
        schema.Insert(0, "$schema", "https://json-schema.org/draft/2020-12/schema");
        schema.Insert(1, "title", "Sharplight verified-model catalog");
        var text = schema.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
        return text.ReplaceLineEndings("\n") + "\n";
    }

    private static JsonNode AddConstraints(JsonSchemaExporterContext context, JsonNode schema)
    {
        if (context.PropertyInfo is not { } property || schema is not JsonObject node)
        {
            return schema;
        }

        string? pattern = (property.DeclaringType.Name, property.Name) switch
        {
            (_, "revision") => "^[0-9a-f]{40}$",
            (_, "sha256") => "^[0-9a-f]{64}$",
            (_, "gitOid") => "^[0-9a-f]{40}$",
            (_, "repository") => "^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$",
            (nameof(CatalogEntry) or nameof(CatalogVariant) or nameof(CatalogCompanion), "id") => "^[a-z0-9][a-z0-9.-]*$",
            (_, "date" or "updated") => "^(\\d{4}-\\d{2}-\\d{2})?$",
            _ => null,
        };
        if (pattern is not null)
        {
            node["pattern"] = pattern;
        }

        if (property.Name == "kind" && property.DeclaringType == typeof(CatalogEntry))
        {
            node["enum"] = new JsonArray([.. CatalogValidator.Kinds.Select(kind => (JsonNode)kind)]);
        }
        else if (property.Name == "format" && property.DeclaringType == typeof(CatalogVariant))
        {
            node["enum"] = new JsonArray([.. CatalogFormats.All.Select(format => (JsonNode)format)]);
        }
        else if (property.Name == "source")
        {
            node["enum"] = new JsonArray([.. CatalogSources.All.Select(source => (JsonNode)source)]);
        }
        else if (property.Name == "role")
        {
            node["enum"] = new JsonArray([.. CatalogValidator.Roles.Select(role => (JsonNode)role)]);
        }

        return node;
    }
}
