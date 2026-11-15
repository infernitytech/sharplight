namespace Sharplight.Templating;

/// <summary>A template filter. <paramref name="value"/> is the filtered value, <paramref name="args"/> the extra arguments; the result is in the template value model.</summary>
public delegate object? FilterFunction(JinjaEnvironment env, object? value, CallArgs args);
