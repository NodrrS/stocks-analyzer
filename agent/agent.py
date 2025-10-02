import json
from dotenv import load_dotenv
from openai import OpenAI
from agent.tools import TOOLS, save_memory, search_web, invoke_model


load_dotenv()

def agent(messages):

    # Initialize the OpenAI client
    client = OpenAI()

    # Make a ChatGPT API call with tool calling
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        tools=TOOLS, # here we pass the tools to the LLM
        messages=messages
    )

    # Get the response from the LLM
    response = completion.choices[0].message

    # Parse the response to get the tool call arguments
    if response.tool_calls:
        # Process each tool call
        for tool_call in response.tool_calls:
            # Get the tool call arguments
            tool_call_arguments = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "save_memory":
                return save_memory(tool_call_arguments["memory"])
            elif tool_call.function.name == "web_search":
                search_results = search_web(tool_call_arguments["query"])
                messages.append({"role": "assistant", "content": f"Here are the search results: {search_results}"})
                return invoke_model(messages)
    else:
        # If there are no tool calls, return the response content
        return response.content