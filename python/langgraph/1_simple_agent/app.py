# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

from api.galileo_streamlit_app import GalileoStreamlitApp
from dotenv import load_dotenv
from galileo.handlers.langchain import GalileoCallback

from agent import AgentFactory

if __name__ == "__main__":
    """Loads the environment variables and runs the agent via the Streamlit App."""
    load_dotenv()
    GalileoStreamlitApp(AgentFactory.create_reference_agent([GalileoCallback()])).run()
