# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

from api.galileo_streamlit_app import GalileoStreamlitApp
from dotenv import load_dotenv
from galileo.handlers.langchain import GalileoCallback
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from agent import Agent
from tools import assess_disruption_risk, check_supplier_compliance
from pinecone_retrieval_tool import PineconeRetrievalTool

if __name__ == "__main__":
    """Loads the environment variables and runs the agent via the Streamlit App."""
    load_dotenv()
    agent = Agent(
            llm=ChatOpenAI(model="gpt-4"),
            tools=[
                TavilySearch(max_results=2),
                assess_disruption_risk,
                check_supplier_compliance,
                PineconeRetrievalTool("supply-chain-information")
            ],
            callbacks=[GalileoCallback()]
        )
    GalileoStreamlitApp(agent).run()
