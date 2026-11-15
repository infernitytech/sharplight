using System.Runtime.CompilerServices;
using Sharplight.Architectures;

namespace Sharplight.Hub.Tests;

/// <summary>
/// Registers the built-in architecture adapters when the test assembly loads: the core loaders only consult the registries, as
/// hosts do through <c>services.AddSharplight()</c>.
/// </summary>
internal static class AdapterRegistration
{
    [ModuleInitializer]
    internal static void Register() => BuiltinAdapters.RegisterAll();
}
