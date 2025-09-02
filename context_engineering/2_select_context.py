#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/9/1 20:01
# @Author  : xinyi
# @File    : 2_select_context.py
# @Description    :
import sys
from rich.console import Console
from rich.pretty import pprint

from utils import llm, save_png, State, format_messages, embeddings
from langgraph.graph import END, START, StateGraph

# Initialize console for rich formatting
console = Console()


def generate_joke(state: State) -> dict[str, str]:
    """Generate an initial joke about the topic.

    Args:
        state: Current state containing the topic

    Returns:
        Dictionary with the generated joke
    """
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


def improve_joke(state: State) -> dict[str, str]:
    """Improve an existing joke by adding wordplay.

    This demonstrates selecting context from state - we read the existing
    joke from state and use it to generate an improved version.

    Args:
        state: Current state containing the original joke

    Returns:
        Dictionary with the improved joke
    """
    print(f"Initial joke: {state['joke']}")

    # Select the joke from state to present it to the LLM
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


# Build the workflow with two sequential nodes
workflow = StateGraph(State)

# Add both joke generation nodes
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)

# Connect nodes in sequence
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", "improve_joke")
workflow.add_edge("improve_joke", END)

# Compile the workflow
chain = workflow.compile()

# Display the workflow visualization
# display(Image(chain.get_graph().draw_mermaid_png()))
save_png(chain.get_graph().draw_mermaid_png(), path="../data/images/chain3.png")
# Execute the workflow to see context selection in action
joke_generator_state = chain.invoke({"topic": "cats"})

# Display the final state with rich formatting
console.print("\n[bold blue]Final Workflow State:[/bold blue]")
pprint(joke_generator_state)

######################### Memory
from langgraph.store.memory import InMemoryStore

# Initialize the memory store
store = InMemoryStore()

# Define namespace for organizing memories
namespace = ("rlm", "joke_generator")

# Store the generated joke in memory
store.put(
    namespace,  # namespace for organization
    "last_joke",  # key identifier
    {"joke": joke_generator_state["joke"]}  # value to store
)

# Select (retrieve) the joke from memory
retrieved_joke = store.get(namespace, "last_joke").value

# Display the retrieved context
console.print("\n[bold green]Retrieved Context from Memory:[/bold green]")
pprint(retrieved_joke)

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# Initialize storage components
checkpointer = InMemorySaver()
memory_store = InMemoryStore()


def generate_joke(state: State, store: BaseStore) -> dict[str, str]:
    """Generate a joke with memory-aware context selection.

    This function demonstrates selecting context from memory before
    generating new content, ensuring consistency and avoiding duplication.

    Args:
        state: Current state containing the topic
        store: Memory store for persistent context

    Returns:
        Dictionary with the generated joke
    """
    # Select prior joke from memory if it exists
    prior_joke = store.get(namespace, "last_joke")
    if prior_joke:
        prior_joke_text = prior_joke.value["joke"]
        print(f"Prior joke: {prior_joke_text}")
    else:
        print("Prior joke: None!")

    # Generate a new joke that differs from the prior one
    prompt = (
        f"Write a short joke about {state['topic']}, "
        f"but make it different from any prior joke you've written: {prior_joke_text if prior_joke else 'None'}"
    )
    msg = llm.invoke(prompt)

    # Store the new joke in memory for future context selection
    store.put(namespace, "last_joke", {"joke": msg.content})

    return {"joke": msg.content}


# Build the memory-aware workflow
workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)

# Connect the workflow
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", END)

# Compile with both checkpointing and memory store
chain = workflow.compile(checkpointer=checkpointer, store=memory_store)

# Execute the workflow with the first thread
config = {"configurable": {"thread_id": "1"}}
joke_generator_state = chain.invoke({"topic": "cats"}, config)

# Get the latest state of the graph
latest_state = chain.get_state(config)

console.print("\n[bold magenta]Latest Graph State:[/bold magenta]")
pprint(latest_state)

# Execute the workflow with a second thread to demonstrate memory persistence
config = {"configurable": {"thread_id": "2"}}
joke_generator_state = chain.invoke({"topic": "cats"}, config)

################ Tools

import math
import types
import uuid

from langgraph.store.memory import InMemoryStore

from langgraph_bigtool import create_agent
from langgraph_bigtool.utils import (
    convert_positional_only_function_to_tool
)

# Collect functions from `math` built-in
all_tools = []
for function_name in dir(math):
    function = getattr(math, function_name)
    if not isinstance(
            function, types.BuiltinFunctionType
    ):
        continue
    # This is an idiosyncrasy of the `math` library
    if tool := convert_positional_only_function_to_tool(
            function
    ):
        all_tools.append(tool)

# Create registry of tools. This is a dict mapping
# identifiers to tool instances.
tool_registry = {
    str(uuid.uuid4()): tool
    for tool in all_tools
}

# Index tool names and descriptions in the LangGraph
# Store. Here we use a simple in-memory store.

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,  # 1024
        "fields": ["description"],
    }
)
for tool_id, tool in tool_registry.items():
    store.put(
        ("tools",),
        tool_id,
        {
            "description": f"{tool.name}: {tool.description}",
        },
    )

# Initialize agent
builder = create_agent(llm, tool_registry)
agent = builder.compile(store=store)
# agent

query = "Use available tools to calculate arc cosine of 0.5."
result = agent.invoke({"messages": query})
format_messages(result['messages'])
