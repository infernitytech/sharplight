namespace Sharplight.ModelFeed;

/// <summary>Kinds of <see cref="PackageDependency"/>.</summary>
public static class DependencyKinds
{
    /// <summary>Installed together with the package; a missing one fails the install.</summary>
    public const string Required = "required";

    /// <summary>Offered with the package; installed when selected.</summary>
    public const string Optional = "optional";
}
