from dotenv import load_dotenv
from fastmcp import FastMCP
from tavily import TavilyClient
import os
from dice_roller import DiceRoller
from huggingface_hub import InferenceClient

#Usecase imports
from elevenlabs import ElevenLabs
from app.rag import retrieve_information
load_dotenv()

mcp = FastMCP("main-server")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))
client_elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
client_huggingface = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about the given query. THIS TOOLD SHOULD NOT BE USED if not related to company policy documents or FAQs."""
    search_results = client.search(query=query)
    return search_results

@mcp.tool()
def roll_dice(notation: str, num_rolls: int = 1) -> str:
    """Roll the dice with the given notation"""
    roller = DiceRoller(notation, num_rolls)
    return str(roller)

"""
Add your own tool here, and then use it through Cursor!
"""
@mcp.tool()
def text_to_speech(text: str, voice_id: str = "JBFqnCBsd6RMkjVDRZzb") -> str:
    """Convert the given text into speech and save as MP3 file"""
    import uuid
    
    try:
        # Generate unique filename
        audio_filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
        
        # Generate audio using ElevenLabs
        audio = client_elevenlabs.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2"
        )
        
        # Save audio to file
        with open(audio_filename, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        
        return f"Audio generated successfully! Saved as: {audio_filename}"
        
    except Exception as e:
        return f"Error generating speech: {str(e)}"

@mcp.tool()
def sentiment_classification(text:str) ->str:
    result = client_huggingface.text_classification(
        text,
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )
    return result[0]["label"]

@mcp.tool()
def retrieve_information_tool(question: str) -> str:
    """Retrieve information from company policy documents and FAQs. THIS TOOLD SHOULD BE USED for questions about policies, returns, procedures, or company information."""
    return retrieve_information.invoke({"question": question})

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)
