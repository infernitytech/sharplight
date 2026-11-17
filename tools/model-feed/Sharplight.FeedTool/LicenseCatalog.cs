using Sharplight.ModelFeed;

namespace Sharplight.FeedTool;

/// <summary>Names, texts and commercial-use flags of the licenses the verified catalog uses (Hub license ids).</summary>
internal static class LicenseCatalog
{
    private static readonly Dictionary<string, (string Name, string Url, bool? Commercial)> Known = new(StringComparer.OrdinalIgnoreCase)
    {
        ["apache-2.0"] = ("Apache License 2.0", "https://www.apache.org/licenses/LICENSE-2.0", true),
        ["mit"] = ("MIT License", "https://opensource.org/license/mit", true),
        ["bsd-3-clause"] = ("BSD 3-Clause License", "https://opensource.org/license/bsd-3-clause", true),
        ["cc-by-4.0"] = ("Creative Commons Attribution 4.0", "https://creativecommons.org/licenses/by/4.0/", true),
        ["cc-by-sa-4.0"] = ("Creative Commons Attribution-ShareAlike 4.0", "https://creativecommons.org/licenses/by-sa/4.0/", true),
        ["cc-by-nc-4.0"] = ("Creative Commons Attribution-NonCommercial 4.0", "https://creativecommons.org/licenses/by-nc/4.0/", false),
        ["cc-by-nc-sa-4.0"] = ("Creative Commons Attribution-NonCommercial-ShareAlike 4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/", false),
        ["openrail"] = ("OpenRAIL", "https://huggingface.co/blog/open_rail", true),
        ["openrail++"] = ("OpenRAIL++-M", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md", true),
        ["creativeml-openrail-m"] = ("CreativeML OpenRAIL-M", "https://huggingface.co/spaces/CompVis/stable-diffusion-license", true),
        ["gemma"] = ("Gemma Terms of Use", "https://ai.google.dev/gemma/terms", true),
        ["llama3.1"] = ("Llama 3.1 Community License", "https://www.llama.com/llama3_1/license/", true),
        ["llama3.2"] = ("Llama 3.2 Community License", "https://www.llama.com/llama3_2/license/", true),
        ["llama3.3"] = ("Llama 3.3 Community License", "https://www.llama.com/llama3_3/license/", true),
    };

    /// <summary>The manifest license for a catalog license id.</summary>
    /// <param name="spdx">Hub license id, or <see langword="null"/>.</param>
    /// <param name="gated">Whether the model is gated.</param>
    /// <returns>The license, or <see langword="null"/> when the catalog records none.</returns>
    public static PackageLicense? Describe(string? spdx, bool gated)
    {
        if (string.IsNullOrWhiteSpace(spdx))
        {
            return gated ? new PackageLicense { Gated = true } : null;
        }

        return Known.TryGetValue(spdx, out var known)
            ? new PackageLicense { Spdx = spdx.ToLowerInvariant(), Name = known.Name, Url = known.Url, CommercialUse = known.Commercial, Gated = gated }
            : new PackageLicense { Spdx = spdx.ToLowerInvariant(), Name = spdx, Gated = gated };
    }
}
