import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from huggingface_hub import InferenceClient

load_dotenv()

mcp = FastMCP("mcp-server")


client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

@mcp.tool()
def sentiment_classification(text:str) ->str:
    result = client.text_classification(
        text,
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )
    return result[0]["label"]

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8002)