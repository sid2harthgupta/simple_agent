# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

from langchain_core.messages import AIMessage
from compound_agent import SharedState


def node_multilingual_combination(state: SharedState):
    """
    Combine all three language responses into a final multilingual response
    """
    # Get responses in all three languages
    english_response = state.get("english_response", "")
    spanish_response = state.get("spanish_response", "")
    hindi_response = state.get("hindi_response", "")

    # If English response is not in state, try to get from messages
    if not english_response:
        for message in reversed(state["messages"]):
            if hasattr(message, 'content') and hasattr(message, 'response_metadata'):
                english_response = message.content
                break

    # Create the multilingual response
    multilingual_response = f"""## 🌍 Supply Chain Analysis - Multilingual Response

        ### 🇺🇸 English Response:
        {english_response}
        
        ---
        
        ### 🇪🇸 Respuesta en Español:
        {spanish_response}
        
        ---
        
        ### 🇮🇳 हिंदी में उत्तर:
        {hindi_response}
        
        ---
        
        *This analysis has been provided in three languages to ensure accessibility across different markets and stakeholders.*
    """

    final_message = AIMessage(content=multilingual_response)

    return {"messages": [final_message]}
