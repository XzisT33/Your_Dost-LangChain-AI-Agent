from fastapi import FastAPI 
from pydantic import BaseModel  
from typing import List 
from langchain_community.tools.tavily_search import TavilySearchResults  
import os  
from langgraph.prebuilt import create_react_agent  
from langchain_groq import ChatGroq  


groq_api_key = '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'  
os.environ["TAVILY_API_KEY"] = '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'  

# Predefined list of supported model names
MODEL_NAMES = [
    "llama3-70b-8192",  
    "mixtral-8x7b-32768"
]

# Initialize the TavilySearchResults tool with a specified maximum number of results.
tool_tavily = TavilySearchResults(max_results=2)  # Allows retrieving up to 2 results


# Combine the TavilySearchResults and ExecPython tools into a list.
tools = [tool_tavily, ]

# FastAPI application setup with a title
app = FastAPI(title='LangGraph AI Agent')

# Define the request schema using Pydantic's BaseModel
class RequestState(BaseModel):
    model_name: str  # Name of the model to use for processing the request
    system_prompt: str  # System prompt for initializing the model
    messages: List[str]  # List of messages in the chat

# Define an endpoint for handling chat requests
@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API endpoint to interact with the chatbot using LangGraph and tools.
    Dynamically selects the model specified in the request.
    """
    if request.model_name not in MODEL_NAMES:
        # Return an error response if the model name is invalid
        return {"error": "Invalid model name. Please select a valid model."}

    # Initialize the LLM with the selected model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=request.model_name)

    # Create a ReAct agent using the selected LLM and tools
    agent = create_react_agent(llm, tools=tools, state_modifier=request.system_prompt)

    # Create the initial state for processing
    state = {"messages": request.messages}

    # Process the state using the agent
    result = agent.invoke(state)  # Invoke the agent (can be async or sync based on implementation)

    # Return the result as the response
    return result

# Run the application if executed as the main script
if __name__ == '__main__':
    import uvicorn  # Import Uvicorn server for running the FastAPI app
    uvicorn.run(app, host='127.0.0.1', port=8000)  # Start the app on localhost with port 8000