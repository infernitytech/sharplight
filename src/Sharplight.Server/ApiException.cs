namespace Sharplight.Server;

/// <summary>An error the API layer reports to the client (the shape depends on the API).</summary>
public sealed class ApiException(int status, string type, string message, string? code = null, string? param = null) : Exception(message)
{
    public int Status { get; } = status;
    public string Type { get; } = type;
    public string? Code { get; } = code;
    public string? Param { get; } = param;
    public static ApiException BadRequest(string message, string? param = null) => new(400, "invalid_request_error", message, null, param);
}
