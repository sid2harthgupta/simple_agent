# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

from api.galileo_streamlit_app import GalileoStreamlitApp
from dotenv import load_dotenv
from galileo.handlers.langchain import GalileoCallback
from langchain_openai import ChatOpenAI

from compound_agent import CompoundAgent

if __name__ == "__main__":
    """Loads the environment variables and runs the agent via the Streamlit App."""
    load_dotenv()
    agent = CompoundAgent(
            llm=ChatOpenAI(model="gpt-4"),
            callbacks=[GalileoCallback()]
        )
    GalileoStreamlitApp(agent).run()
