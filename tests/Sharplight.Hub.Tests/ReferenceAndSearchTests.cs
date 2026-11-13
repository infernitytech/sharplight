namespace Sharplight.Hub.Tests;

public sealed class ReferenceAndSearchTests
{
    [Theory]
    [InlineData("Qwen/Qwen3-0.6B-GGUF", "Qwen/Qwen3-0.6B-GGUF", null)]
    [InlineData("Qwen/Qwen3-0.6B-GGUF:Q8_0", "Qwen/Qwen3-0.6B-GGUF", "Q8_0")]
    [InlineData("hf.co/unsloth/Qwen3-0.6B-GGUF:Q4_K_M", "unsloth/Qwen3-0.6B-GGUF", "Q4_K_M")]
    [InlineData("https://huggingface.co/Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", null)]
    public void ParsesReferences(string text, string repository, string? tag)
    {
        Assert.True(ModelReference.TryParse(text, out var reference));
        Assert.Equal(repository, reference.Repository);
        Assert.Equal(tag, reference.Tag);
    }

    [Theory]
    [InlineData("qwen3-0.6b")]
    [InlineData("qwen3-0.6b:q8_0")]
    [InlineData("a/b/c")]
    [InlineData("owner/")]
    [InlineData("owner/name:")]
    [InlineData("../etc")]
    public void RejectsNonRepositories(string text)
    {
        Assert.False(ModelReference.TryParse(text, out _));
    }

    [Fact]
    public void ParsesSearchResults()
    {
        const string Json = """
            [
              {"id":"Qwen/Qwen3-0.6B-GGUF","downloads":215881,"likes":113,"gated":false,"library_name":"gguf",
               "gguf":{"total":596049920,"architecture":"qwen3"},
               "siblings":[{"rfilename":"Qwen3-0.6B-Q8_0.gguf"},{"rfilename":"Qwen3-0.6B-Q4_K_M.gguf"},{"rfilename":"README.md"}]},
              {"id":"meta-llama/Llama-3.2-1B","downloads":632512,"likes":2776,"gated":"manual","pipeline_tag":"text-generation",
               "safetensors":{"parameters":{"BF16":1235814400},"total":1235814400},
               "siblings":[{"rfilename":"config.json"},{"rfilename":"model.safetensors"},{"rfilename":"original/consolidated.00.pth"}]},
              {"modelId":"x/onnx-model","downloads":1,"likes":0,"siblings":[{"rfilename":"onnx/model.onnx"},{"rfilename":"onnx/model_q4.onnx"}]}
            ]
            """;

        var results = HubJson.ParseSearch(Json);

        Assert.Equal(3, results.Count);
        var gguf = results[0];
        Assert.Equal("Qwen/Qwen3-0.6B-GGUF", gguf.Id);
        Assert.Equal(215881, gguf.Downloads);
        Assert.Equal(113, gguf.Likes);
        Assert.False(gguf.Gated);
        Assert.Equal(596049920, gguf.Parameters);
        Assert.Equal([ModelFormat.Gguf], gguf.Formats);
        Assert.Equal(["Q4_K_M", "Q8_0"], gguf.Variants);

        var gated = results[1];
        Assert.True(gated.Gated);
        Assert.Equal("text-generation", gated.Pipeline);
        Assert.Equal(1235814400, gated.Parameters);
        Assert.Equal([ModelFormat.Safetensors], gated.Formats);
        Assert.Empty(gated.Variants);

        var onnx = results[2];
        Assert.Equal("x/onnx-model", onnx.Id);
        Assert.Null(onnx.Parameters);
        Assert.Equal(["q4"], onnx.Variants);
    }

    [Fact]
    public async Task SearchSendsQueryAndFormatFilter()
    {
        using var hub = new FakeHub { Intercept = _ => new HttpResponseMessage(System.Net.HttpStatusCode.OK) { Content = new StringContent("[]") } };
        using var http = hub.CreateClient();
        var client = new HubClient(http, hub.Options());

        var results = await client.SearchAsync(new HubSearchQuery { Text = "qwen 3", Format = ModelFormat.Gguf, Limit = 5 }, TestContext.Current.CancellationToken);

        Assert.Empty(results);
        var query = Uri.UnescapeDataString(Assert.Single(hub.Requests).RequestUri!.Query);
        Assert.Contains("search=qwen 3", query, StringComparison.Ordinal);
        Assert.Contains("filter=gguf", query, StringComparison.Ordinal);
        Assert.Contains("limit=5", query, StringComparison.Ordinal);
        Assert.Contains("expand[]=siblings", query, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RepositoryListingPinsCommitAndReadsLfsChecksums()
    {
        using var hub = new FakeHub();
        var repository = hub.Add("owner/model", commit: "c0ffee").WithSize("big.gguf", 5000).With("README.md", "hi");
        using var http = hub.CreateClient();
        var client = new HubClient(http, hub.Options());

        var resolved = await client.GetRepositoryAsync("owner/model", "main", TestContext.Current.CancellationToken);

        Assert.Equal("c0ffee", resolved.Commit);
        var big = resolved.Files.Single(file => file.Path == "big.gguf");
        Assert.Equal(5000, big.Size);
        Assert.Equal(64, big.Sha256!.Length);
        Assert.Null(resolved.Files.Single(file => file.Path == "README.md").Sha256);
        Assert.EndsWith("/owner/model/resolve/c0ffee/big.gguf", client.GetFileUri(resolved, "big.gguf").AbsoluteUri, StringComparison.Ordinal);
        Assert.Equal(repository.Commit, resolved.Commit);
    }

    [Fact]
    public async Task MissingRepositoryIsAClearError()
    {
        using var hub = new FakeHub();
        using var http = hub.CreateClient();
        var client = new HubClient(http, hub.Options());

        var error = await Assert.ThrowsAsync<HubException>(() => client.GetRepositoryAsync("owner/missing", "main", TestContext.Current.CancellationToken));

        Assert.Contains("not found", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("""{"id":"o/m","sha":"abc","gguf":{"architecture":"qwen3"},"config":{"model_type":"other"}}""", "qwen3")]
    [InlineData("""{"id":"o/m","sha":"abc","config":{"model_type":"llama"}}""", "llama")]
    [InlineData("""{"id":"o/m","sha":"abc","gated":"auto"}""", null)]
    public void RevisionReportsTheModelFamily(string json, string? architecture)
    {
        var info = HubJson.ParseRevision(json);

        Assert.Equal("abc", info.Commit);
        Assert.Equal(architecture, info.Architecture);
    }
}
