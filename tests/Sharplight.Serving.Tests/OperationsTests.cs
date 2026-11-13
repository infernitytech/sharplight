using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Sharplight.Llm.Serving;
using Sharplight.AspNetCore;

namespace Sharplight.Serving.Tests;

/// <summary>Liveness / readiness probes, tenant identification and the bounded fair-share bookkeeping.</summary>
public class OperationsTests
{
    [Fact]
    public async Task LiveAndReadyProbes_ReflectDraining()
    {
        await using var h = await ApiHost.Create(o => o.ApiKeys = ["secret"]);
        using var live = await h.Client.GetAsync("/health/live", TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.OK, live.StatusCode);   // open without a key, like /health
        using var ready = await h.Client.GetAsync("/health/ready", TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.OK, ready.StatusCode);
        Assert.Equal("ready", JsonDocument.Parse(await ready.Content.ReadAsStringAsync(TestContext.Current.CancellationToken)).RootElement.GetProperty("status").GetString());

        await h.Engine.DrainAsync(TimeSpan.FromSeconds(5));
        using var draining = await h.Client.GetAsync("/health/ready", TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.ServiceUnavailable, draining.StatusCode);
        using var stillLive = await h.Client.GetAsync("/health/live", TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.OK, stillLive.StatusCode);
    }

    [Fact]
    public void Tenant_IsUserElseHashedKey()
    {
        var http = new DefaultHttpContext();
        Assert.Null(Tenants.Of(http, null));
        http.Request.Headers.Authorization = "Bearer sk-secret";
        string? byKey = Tenants.Of(http, null);
        Assert.StartsWith("key:", byKey);
        Assert.DoesNotContain("secret", byKey);
        Assert.Equal("user:alice", Tenants.Of(http, "alice"));
        var other = new DefaultHttpContext();
        other.Request.Headers["x-api-key"] = "sk-secret";
        Assert.Equal(byKey, Tenants.Of(other, null));
    }

    [Fact]
    public void FairShare_ForgetsIdleTenants()
    {
        var q = new RequestQueue(TimeSpan.FromTicks(1));   // usage decays immediately
        for (int i = 0; i < 5000; i++)
        {
            q.Charge("t" + i, 10);
        }

        Assert.True(q.TenantCount <= 4096, $"{q.TenantCount} tenants tracked");
    }
}
