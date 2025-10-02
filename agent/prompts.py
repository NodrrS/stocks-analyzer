from agent.tools import load_memories
from datetime import datetime

def get_system_prompt(user_prompt):

    memories = load_memories(user_prompt)

    return f"""
    - You are a helpful assistant with memory and web search capabilities.
    - Use the save_memory function to save memories to the vector database. 
    - You should only use this function to save memories relative to the user's preferences.
    - Use the web_search function to search the web for information that requires real-time data.
    
    CURRENT DATETIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    MEMORIES:
    {memories}
    """