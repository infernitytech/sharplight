namespace Sharplight.Templating;

/// <summary>A template test (<c>x is name</c>). <paramref name="value"/> is the tested value, <paramref name="args"/> the extra arguments.</summary>
public delegate bool TestFunction(JinjaEnvironment env, object? value, CallArgs args);
