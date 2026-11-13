using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;

namespace Sharplight.Server;

internal static class HttpJson
{
    public static readonly JsonSerializerOptions Options = new() { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    public static string ToJson(JsonNode n) => n.ToJsonString(Options);

    public static async Task WriteAsync(HttpContext http, JsonNode body, int? status = null)
    {
        if (status is int s)
        {
            http.Response.StatusCode = s;
        }

        http.Response.ContentType = "application/json";
        await http.Response.WriteAsync(ToJson(body), Encoding.UTF8, http.RequestAborted);
    }

    /// <summary>Reads and parses the JSON request body, enforcing the size limit.</summary>
    public static async Task<JsonDocument> ReadBodyAsync(HttpContext http, long maxBytes)
    {
        if (http.Request.ContentLength > maxBytes)
        {
            throw new ApiException(413, "request_too_large", $"The request body exceeds the limit of {maxBytes} bytes.");
        }

        if (http.Features.Get<IHttpMaxRequestBodySizeFeature>() is { IsReadOnly: false } f)
        {
            f.MaxRequestBodySize = maxBytes;
        }

        using var ms = new MemoryStream();
        var buf = new byte[81920];
        int n;
        while ((n = await http.Request.Body.ReadAsync(buf, http.RequestAborted)) > 0)
        {
            ms.Write(buf, 0, n);
            if (ms.Length > maxBytes)
            {
                throw new ApiException(413, "request_too_large", $"The request body exceeds the limit of {maxBytes} bytes.");
            }
        }
        try
        {
            var doc = JsonDocument.Parse(ms.ToArray());
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
            {
                throw ApiException.BadRequest("The request body must be a JSON object.");
            }

            return doc;
        }
        catch (JsonException ex) { throw ApiException.BadRequest("Invalid JSON: " + ex.Message); }
    }
}
