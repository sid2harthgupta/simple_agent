import uuid

import streamlit as st
from dotenv import load_dotenv
from galileo import galileo_context
from galileo.handlers.langchain import GalileoCallback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from agent import Agent
from tools import check_supplier_compliance, assess_disruption_risk

load_dotenv()


def display_chat_history(agent: Agent):
    """Display all messages from the agent's conversation history."""
    messages = agent.get_conversation_history()

    if not messages:
        return

    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            # Skip empty AI messages (tool decision messages)
            if message.content and message.content.strip():
                with st.chat_message("assistant"):
                    st.markdown(message.content)


def initialize_agent(thread_id: str):
    """Initialize the agent with proper configuration."""
    with st.spinner("Initializing agent..."):
        try:
            agent = Agent(
                llm=ChatOpenAI(model="gpt-4"),
                tools=[TavilySearch(max_results=2), assess_disruption_risk, check_supplier_compliance],
                thread_id=thread_id,
                callbacks=[GalileoCallback()],
                # history=[SystemMessage(content="You are a helpful supply chain assistant.")]
            )
            return agent

        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.stop()


def get_session_id():
    """Generate a unique session ID for this Streamlit session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return st.session_state.session_id


def new_conversation():
    """Start a new conversation by creating a new thread ID."""
    new_thread_id = str(uuid.uuid4())[:8]
    galileo_context.start_session(name="", external_id=new_thread_id)
    st.session_state.agent = initialize_agent(new_thread_id)


def main():
    """Main function for the Streamlit app."""

    # Initialize session ID
    session_id = get_session_id()

    st.title("Supply Chain Assistant")

    # Initialize the agent (only once per session)
    if "agent" not in st.session_state:
        galileo_context.start_session(name="", external_id=session_id)
        thread_id = session_id
        st.session_state.agent = initialize_agent(thread_id)

    # Sidebar
    with st.sidebar:
        st.markdown("### Capabilities")
        st.markdown("* Web search")
        st.markdown("* Tool calling")
        st.markdown("* Memory")
        st.divider()
        if st.button("Start new conversation"):
            new_conversation()
            st.rerun()

    # Always show example queries
    st.markdown("### Example queries:")
    examples = [
        "Who has won the most men's singles tennis matches?",
        "What is the compliance status for SUP001?",
        "What is the square root of 34 multiplied by 5?",
    ]

    for i, example in enumerate(examples):
        col1, col2 = st.columns([1, 3])
        with col2:
            st.code(example, language=None)

    # Display chat history
    display_chat_history(st.session_state.agent)

    # Chat input
    user_input = st.chat_input("How can I help you?...")

    # Process user input when submitted
    if user_input:
        # Display the user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Show thinking spinner while processing
        with st.spinner("Thinking..."):
            try:
                # Process the user input through the agent
                st.session_state.agent.invoke(user_input)

            except Exception as e:
                st.error(f"Error: {str(e)}")

        # Rerun to update the display with the full conversation
        st.rerun()


if __name__ == "__main__":
    main()
