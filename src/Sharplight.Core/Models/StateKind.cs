namespace Sharplight.Models;

/// <summary>Kind of persistent state; lets runtimes manage e.g. context shifting or snapshots per kind.</summary>
public enum StateKind { KeyCache, ValueCache, Recurrent, Other }
