# Copyright (c) 2025 Galileo Technologies, Inc. All rights reserved.

import uuid
import warnings
from typing import Annotated, Dict, List
import re
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from api.base_agent import BaseAgent
from api.base_message import BaseMessage
from common.utils import LangGraphUtils
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

warnings.filterwarnings("ignore", category=SyntaxWarning, module="langchain_tavily")


class AgentState(TypedDict):
    """
    In LangGraph, every node updates a shared graph state. The state is the input to any node whenever it is invoked.
    Below, we will define a state dict to contain the task, plan, steps, and other variables.
    """
    # messages: Annotated[List, add_messages]
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str


planner_prompt = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Tools can be one of the following:
(1) Google[input]: Worker that searches results from Google. Useful when you need to find short
and succinct answers about a specific topic. The input should be a search query.
(2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
world knowledge and common sense. Prioritize it when you are confident in solving the problem
yourself. Input can be any instruction.

For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?
Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
with Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Find out the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked. #E3 = Calculator[(2 ∗ #E2 − 10) − 8]

Begin! 
Describe your plans with rich details. Each Plan should be followed by only one #E.

Task: {task}"""

# Planner node
model = ChatOpenAI(model="gpt-4o")
# Regex to match expressions of the form E#... = ...[...]
regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
prompt_template = ChatPromptTemplate.from_messages([("user", planner_prompt)])
planner = prompt_template | model


def get_plan(state: AgentState):
    task = state["task"]
    result = planner.invoke({"task": task})
    # Find all matches in the sample text
    matches = re.findall(regex_pattern, result.content)
    return {"steps": matches, "plan_string": result.content}


# Executor node
search = TavilySearchResults()
def _get_current_task(state: AgentState):
    if "results" not in state or state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


def tool_execution(state: AgentState):
    """Worker node that executes the tools of a given plan."""
    _step = _get_current_task(state)
    _, step_name, tool, tool_input = state["steps"][_step - 1]
    _results = (state["results"] or {}) if "results" in state else {}
    for k, v in _results.items():
        tool_input = tool_input.replace(k, v)
    if tool == "Google":
        result = search.invoke(tool_input)
    elif tool == "LLM":
        result = model.invoke(tool_input)
    else:
        raise ValueError
    _results[step_name] = str(result)
    return {"results": _results}


# Solver
solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""


def solve(state: AgentState):
    plan = ""
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = (state["results"] or {}) if "results" in state else {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
    prompt = solve_prompt.format(plan=plan, task=state["task"])
    result = model.invoke(prompt)
    return {"result": result.content}


# Define graph
def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"

graph = StateGraph(AgentState)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("solve", solve)
graph.add_edge("plan", "tool")
graph.add_edge("solve", END)
graph.add_conditional_edges("tool", _route)
graph.add_edge(START, "plan")

app = graph.compile()

task = "what is the exact hometown of the 2024 mens australian open winner"

for s in app.stream({"task": task}):
    print(s)
    print("---")

print(s["solve"]["result"])


class Agent(BaseAgent):
    """LangGraph wrapper for stateful agents with tool support and conversation memory.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_tavily import TavilySearch
        >>> from galileo.handlers.langchain import GalileoCallback
        >>> from api.base_message import BaseMessage, BaseMessageType
        >>>
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> tools = [TavilySearch(max_results=3)]
        >>> callbacks = [GalileoCallback()]
        >>> agent = Agent(llm, tools, callbacks)
        >>> response = agent.invoke([BaseMessage(message_type=BaseMessageType.HumanMessage, content="What are supply chain risks?")])
    """

    def __init__(
            self,
            llm: BaseChatModel,
            tools: List[BaseTool],
            callbacks: List
    ) -> None:
        """Initialize agent with LangGraph-managed state.

        Args:
            llm: Chat model for generating responses.
            tools: Available tools for the agent.
            callbacks: Monitoring/logging callbacks.
        """
        self.tools = tools
        self.llm_with_tools = llm.bind_tools(tools)
        self.callbacks = callbacks
        self.config = self._create_config(callbacks)

        self.graph = self._build_graph()

    @property
    def name(self) -> str:
        return "Supply Chain Agent"

    @property
    def capabilities(self) -> List[str]:
        return ["Web search", "Tool calling", "Memory", "RAG"]

    @property
    def example_queries(self) -> List[str]:
        return [
            "Give me a short intro to the fundamentals of supply chain.",
            "What is the compliance status for SUP001?",
            "Who has won the most men's singles tennis matches?",
            "What is the square root of 34 multiplied by 5?",
        ]

    def reset(self) -> None:
        self.config = self._create_config(self.callbacks)

    def invoke(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        assert len(messages) == 1
        result = self.graph.invoke(
            {"messages": [LangGraphUtils.to_langchain_message(messages[0])]},
            config=self.config
        )
        return [LangGraphUtils.to_base_message(result["messages"][-1])]

    def get_message_history(self) -> List[BaseMessage]:
        """Get current conversation history."""
        current_state = self.graph.get_state(config=self.config)
        messages = [m for m in current_state.values.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
        return [LangGraphUtils.to_base_message(message) for message in messages]

    def _invoke_chatbot(self, state: AgentState) -> Dict[str, List]:
        """Internal chatbot node implementation."""
        message = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [message]}

    def _build_graph(self) -> CompiledStateGraph:
        """Build LangGraph with automatic state persistence."""
        graph_builder = StateGraph(AgentState)

        # Nodes
        graph_builder.add_node("chatbot", self._invoke_chatbot)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))

        # Edges
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")

        return graph_builder.compile(checkpointer=MemorySaver())

    @staticmethod
    def _create_config(callbacks: List) -> dict:
        return {
            "configurable": {"thread_id": str(uuid.uuid4())[:8]},
            "callbacks": callbacks
        }
