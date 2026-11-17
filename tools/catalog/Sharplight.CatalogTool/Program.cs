using Sharplight.CatalogTool;

return await new CatalogCommands().Build().Parse(args).InvokeAsync().ConfigureAwait(false);
