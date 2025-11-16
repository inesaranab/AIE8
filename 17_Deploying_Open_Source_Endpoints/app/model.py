from langchain_together import ChatTogether
from langchain_openai import ChatOpenAI


def get_chat_model(model_name: str = "gpt-oss", temperature: float = 0):
    """
    Get a chat model instance based on the model name.
    
    Args:
        model_name: Name of the model to use. 
                   - If contains "gpt-4" or "gpt-3.5", uses OpenAI
                   - Otherwise uses Together AI
                   Default: "gpt-oss"
        temperature: Temperature for model generation. Default: 0
    
    Returns:
        Configured chat model instance (ChatOpenAI or ChatTogether)
    
    Examples:
        >>> model = get_chat_model("openai/gpt-oss-20b")  # Uses Together AI
        >>> model = get_chat_model("gpt-4.1-mini")        # Uses OpenAI
        >>> model = get_chat_model()                       # Uses gpt-oss (Together AI)
    """
    if "gpt-4.1-mini" in model_name or "gpt-4.1-nano" in model_name:
        return ChatOpenAI(model=model_name, temperature=temperature)
    else:
        return ChatTogether(model=model_name, temperature=temperature)

