using Sharplight.DesignTool;

return await DesignToolCommands.Build().Parse(args).InvokeAsync().ConfigureAwait(false);
