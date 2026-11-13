namespace Sharplight.Llm.Serving.Chat;

public enum ChatFinishReason { Stop, Length, ToolCalls, Cancelled, Error, Timeout }
