using Sharplight.FeedTool;

return await new FeedCommands().Build().Parse(args).InvokeAsync().ConfigureAwait(false);
