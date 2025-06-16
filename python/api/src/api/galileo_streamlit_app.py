# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

import uuid

import streamlit as st
from api.base_agent import BaseAgent
from api.base_message import BaseMessageType, BaseMessage
from galileo import galileo_context


class GalileoStreamlitApp:
    """Streamlit app that works for any BaseAgent.
    Agents from any framework can be wrapped into a
    BaseAgent to work with this streamlit app.
    """
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    def run(self):
        st.title("Supply Chain Assistant")

        # Initialize the agent (only once per session)
        if "agent" not in st.session_state:
            session_id = str(uuid.uuid4())[:8]
            st.session_state.session_id = session_id
            galileo_context.start_session(name="", external_id=session_id)
            st.session_state.agent = self.agent

        # Sidebar
        with st.sidebar:
            st.markdown("### Capabilities")
            for capability in st.session_state.agent.capabilities:
                st.markdown(f"* {capability}")
            st.divider()
            if st.button("Start new conversation"):
                self.new_conversation()
                st.rerun()

        # Always show example queries
        st.markdown("### Example queries:")
        for example in st.session_state.agent.example_queries:
            col1, col2 = st.columns([1, 3])
            with col2:
                st.code(example, language=None)

        # Display chat history
        self.display_chat_history()

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
                    st.session_state.agent.invoke(
                        [BaseMessage(message_type=BaseMessageType.HumanMessage, content=user_input)]
                    )

                except Exception as e:
                    st.error(f"Error: {str(e)}")

            # Rerun to update the display with the full conversation
            st.rerun()

    @staticmethod
    def display_chat_history():
        """Display the human and AI messages from the agent's conversation history.
        Currently, does not include other message types like System and Tool.
        """
        messages = st.session_state.agent.get_message_history()

        if not messages:
            return

        for message in messages:
            if message.message_type == BaseMessageType.HumanMessage:
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif message.message_type == BaseMessageType.AiMessage:
                # Skip empty AI messages (tool decision messages)
                if message.content and message.content.strip():
                    with st.chat_message("assistant"):
                        st.markdown(message.content)
            # Skip all other message types for now

    @staticmethod
    def new_conversation():
        """Start a new conversation by creating a new session ID."""
        new_session_id = str(uuid.uuid4())[:8]
        st.session_state.session_id = new_session_id
        galileo_context.start_session(name="", external_id=new_session_id)
        st.session_state.agent.reset()
