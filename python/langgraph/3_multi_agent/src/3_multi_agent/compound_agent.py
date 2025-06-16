# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.
import uuid
from typing import List

from api.base_agent import BaseAgent
from api.base_message import BaseMessage
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from supply_chain_agent.agent import SupplyChainAgent
from financial_agent.financial_agent import FinancialAgent
from planner_agent.planner_agent import PlannerAgent
from translation_agent.translation_agent import TranslationAgent
from nodes import node_multilingual_combination

from typing import TypedDict, Annotated, Optional

from langgraph.graph.message import add_messages


class SharedState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: Optional[str]
    english_response: Optional[str]
    spanish_response: Optional[str]
    hindi_response: Optional[str]


class CompoundAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel, callbacks: List):
        self.planner = PlannerAgent(llm, callbacks)
        self.supply_chain_agent = SupplyChainAgent(llm, callbacks=callbacks)
        self.financial_agent = FinancialAgent(llm, callbacks=callbacks)
        self.spanish_translator = TranslationAgent(llm, callbacks=callbacks, language="spanish")
        self.hindi_translator = TranslationAgent(llm, callbacks=callbacks, language="hindi")

    @property
    def name(self) -> str:
        return 'agent'

    @property
    def capabilities(self) -> List[str]:
        return []

    @property
    def example_queries(self) -> List[str]:
        return []

    def reset(self) -> None:
        pass

    def invoke(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        pass

    def get_message_history(self) -> List[BaseMessage]:
        pass

    def _build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(SharedState)

        # Add the intent classifier
        graph_builder.add_node("planner", self.planner)

        # Add the compiled subgraphs as nodes with callback wrappers
        graph_builder.add_node("supply_chain_agent", supply_chain_with_callbacks)
        graph_builder.add_node("financial_agent", financial_with_context)

        # Add synthesis node
        graph_builder.add_node("synthesize_followup", synthesize_followup)

        # Add translation nodes
        graph_builder.add_node("spanish_translation", spanish_translation)
        graph_builder.add_node("hindi_translation", hindi_translation)

        # Add final combination node
        graph_builder.add_node("multilingual_combination", node_multilingual_combination)

        # Set entry point
        graph_builder.add_edge(START, "intent_classifier")

        # Add conditional routing from intent classifier
        graph_builder.add_conditional_edges(
            "intent_classifier",
            route_to_agents,
            {
                "supply_chain_agent": "supply_chain_agent",
                "financial_agent": "financial_agent",
                "both_agents": "supply_chain_agent"  # Start with supply chain for collaboration
            }
        )

        # For collaborative workflows: supply_chain -> financial -> synthesis
        def after_supply_chain(state: SharedState):
            if state.get("next_agent") == "both_agents":
                return "financial_agent"
            else:
                return "synthesize_followup"

        graph_builder.add_conditional_edges(
            "supply_chain_agent",
            after_supply_chain,
            {
                "financial_agent": "financial_agent",
                "synthesize_followup": "synthesize_followup"
            }
        )

        # Financial agent always goes to synthesis
        graph_builder.add_edge("financial_agent", "synthesize_followup")

        graph_builder.add_edge("synthesize_followup", "spanish_translation")
        graph_builder.add_edge("synthesize_followup", "hindi_translation")

        # Both translation nodes go to final combination
        graph_builder.add_edge("spanish_translation", "multilingual_combination")
        graph_builder.add_edge("hindi_translation", "multilingual_combination")

        # Final combination goes to end
        graph_builder.add_edge("multilingual_combination", END)

        return graph_builder.compile(checkpointer=MemorySaver())

    @staticmethod
    def _create_config(callbacks: List) -> dict:
        return {
            "configurable": {"thread_id": str(uuid.uuid4())[:8]},
            "callbacks": callbacks
        }




def route_to_agents(state: State):
    """
    Determine which path to take based on classification
    """
    next_agent = state.get("next_agent", "supply_chain_agent")

    if next_agent == "both_agents":
        return "supply_chain_agent"  # Start with supply chain, then go to financial
    else:
        return next_agent


def financial_with_context(state: State):
    """
    Wrapper for financial agent that adds context from supply chain agent
    when running in collaborative mode
    """
    # Check if we have supply chain results to add as context
    supply_chain_result = None
    for message in reversed(state["messages"]):
        if hasattr(message, 'content') and "supply chain" in message.content.lower():
            supply_chain_result = message.content
            break

    config = {"configurable": {"thread_id": "financial-subgraph"}}
    if _subgraph_callbacks:
        config["callbacks"] = _subgraph_callbacks
    financial_agent = get_financial_agent()

    if supply_chain_result and state.get("next_agent") == "both_agents":
        # Add context message for financial agent
        context_message = SystemMessage(content=f"""
        Context from Supply Chain Analysis:
        {supply_chain_result}

        Please provide financial analysis that complements the supply chain analysis above.
        Focus on cost implications, ROI calculations, and financial risks related to the supply chain recommendations.
        Consider the operational factors mentioned in the supply chain analysis.
        """)

        # Add context to messages
        messages_with_context = state["messages"] + [context_message]
        modified_state = {**state, "messages": messages_with_context}

        return financial_agent.invoke(modified_state, config)
    else:
        return financial_agent.invoke(state, config)


def supply_chain_with_callbacks(state: State):
    """
    Wrapper for supply chain agent to ensure callbacks are passed through
    """
    supply_chain_agent = get_supply_chain_agent()
    config = {"configurable": {"thread_id": "supply-chain-subgraph"}}

    if _subgraph_callbacks:
        config["callbacks"] = _subgraph_callbacks

    return supply_chain_agent.invoke(state, config)


def get_modular_multi_agent():
    """
    Create the main orchestrator graph using existing agent subgraphs as nodes
    """
    # Build the main orchestrator graph



class ModularMultiAgentOrchestrator:
    """
    Orchestrator that uses existing agent subgraphs as building blocks
    """

    def __init__(self, callbacks=None):
        set_subgraph_callbacks(callbacks)
        self.graph = get_modular_multi_agent()
        self.config = {"configurable": {"thread_id": "modular-multi-agent"}}

        if callbacks:
            self.config["callbacks"] = callbacks

    def process_query(self, user_query: str) -> str:
        """Process a query through the modular multi-agent system"""
        initial_state = {"messages": [HumanMessage(content=user_query)]}
        result = self.graph.invoke(initial_state, self.config)

        # Return the last message content
        if result["messages"]:
            return result["messages"][-1].content
        return "No response generated"

    def get_routing_decision(self, user_query: str) -> Dict[str, Any]:
        """Get routing decision without full execution"""
        # Just run the intent classifier
        state = {"messages": [HumanMessage(content=user_query)]}
        classifier_result = intent_classifier(state)

        next_agent = classifier_result.get("next_agent", "supply_chain_agent")

        return {
            "primary_agent": next_agent,
            "requires_collaboration": next_agent == "both_agents",
            "execution_order": ["supply_chain", "financial"] if next_agent == "both_agents" else [
                next_agent.replace("_agent", "")]
        }






