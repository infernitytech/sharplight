using Microsoft.AspNetCore.Http;
using Sharplight.AspNetCore;

namespace Sharplight.Serving.Tests;

/// <summary>How the request context maps to the engine: fair-share tenant, cache partition and token limits.</summary>
public sealed class RequestContextTests
{
    private static DefaultHttpContext Request(SharplightRequestContext context)
    {
        var http = new DefaultHttpContext();
        http.Request.Headers.Authorization = "Bearer some-key";
        http.Features.Set(new SharplightRequestFeature(context, new SharplightEndpointOptions()));
        return http;
    }

    [Fact]
    public void TheTenantComesFromTheResolverElseFromTheScopeAndTheUser()
    {
        Assert.Equal("fair-share-id", SharplightServerContext.TenantOf(Request(new SharplightRequestContext { Scope = "space-1", Tenant = "fair-share-id" }), "u"));
        Assert.Equal("space-1/user:u", SharplightServerContext.TenantOf(Request(new SharplightRequestContext { Scope = "space-1" }), "u"));
        Assert.StartsWith("space-1/key:", SharplightServerContext.TenantOf(Request(new SharplightRequestContext { Scope = "space-1" }), null), StringComparison.Ordinal);
        Assert.Equal("user:u", SharplightServerContext.TenantOf(Request(new SharplightRequestContext()), "u"));
    }

    [Fact]
    public void TheScopeIsTheCachePartition()
    {
        Assert.Equal("space-1", SharplightServerContext.CachePartition(Request(new SharplightRequestContext { Scope = "space-1" })));
        Assert.Null(SharplightServerContext.CachePartition(Request(new SharplightRequestContext())));
        Assert.Null(SharplightServerContext.CachePartition(new DefaultHttpContext()));
    }

    [Fact]
    public void TokenLimitsCutRequests()
    {
        var limited = Request(new SharplightRequestContext { Defaults = new SharplightRequestDefaults { MaxTokens = 100, MaxTokensLimit = 500 } });
        Assert.Equal(100, SharplightServerContext.MaxTokens(limited, null, 0));
        Assert.Equal(300, SharplightServerContext.MaxTokens(limited, 300, 0));
        Assert.Equal(500, SharplightServerContext.MaxTokens(limited, 9000, 0));
        Assert.Equal(16, SharplightServerContext.MaxTokens(new DefaultHttpContext(), null, 16));
    }
}
