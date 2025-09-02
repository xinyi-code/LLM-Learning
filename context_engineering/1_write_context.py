#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/9/1 14:56
# @Author  : xinyi
# @File    : 1_write_context.py
# @Description    :

############################ Scratchpad

import sys
import os
import getpass
from rich.console import Console
from rich.pretty import pprint
from utils import llm, save_png, State

# Initialize console for rich formatting
console = Console()

from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph

# Set up environment and initialize model




def generate_joke(state: State) -> dict[str, str]:
    """Generate a joke about the specified topic.

    This node reads the topic from state and generates a joke,
    then writes the joke back to state.

    Args:
        state: Current state containing the topic

    Returns:
        Dictionary with the generated joke
    """
    # Read the topic key from state, and pass it in the LLM prompt
    msg = llm.invoke(f"Write a short joke about {state['topic']}")

    # Write context (our joke) to a field in state
    return {"joke": msg.content}

# Initialize StateGraph with our state schema
workflow = StateGraph(State)

# Add our joke generation node to the workflow
workflow.add_node("generate_joke", generate_joke)

# Connect the node to workflow start and end
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", END)

# Compile the workflow into an executable graph
chain = workflow.compile()

# Display the workflow visualization
img_data = chain.get_graph().draw_mermaid_png()
save_png(img_data, path="../data/images/chain.png")
# display(Image(chain.get_graph().draw_mermaid_png())) # jupyter

# Execute the workflow with a specific topic
joke_generator_state = chain.invoke({"topic": "cats"})

# Display the resulting state with rich formatting
console.print("\n[bold blue]Joke Generator State:[/bold blue]")
pprint(joke_generator_state)


############### Memory
from langgraph.store.memory import InMemoryStore

# Initialize the in-memory store for long-term memory
store = InMemoryStore()

# Define namespace as a tuple (user_id, application_context)
namespace = ("rlm", "joke_generator")

# Write context as a key-value pair to the namespace
store.put(
    namespace,                             # namespace for organizing data
    "last_joke",                          # key for this specific piece of data
    {"joke": joke_generator_state["joke"]} # value to store
)

# Search the namespace to view all stored items
stored_items = list(store.search(namespace))

# Display the stored items with rich formatting
console.print("\n[bold green]Stored Items in Memory:[/bold green]")
pprint(stored_items)

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# Initialize storage components
checkpointer = InMemorySaver()  # For thread-level state persistence
memory_store = InMemoryStore()  # For cross-thread memory storage


def generate_joke(state: State, store: BaseStore) -> dict[str, str]:
    """Generate a joke with memory awareness.

    This enhanced version checks for existing jokes in memory
    before generating new ones.

    Args:
        state: Current state containing the topic
        store: Memory store for persistent context

    Returns:
        Dictionary with the generated joke
    """
    # Check if there's an existing joke in memory
    existing_jokes = list(store.search(namespace))
    if existing_jokes:
        existing_joke = existing_jokes[0].value
        print(f"Existing joke: {existing_joke}")
    else:
        print("Existing joke: No existing joke")

    # Generate a new joke based on the topic
    msg = llm.invoke(f"Write a short joke about {state['topic']}")

    # Store the new joke in long-term memory
    store.put(namespace, "last_joke", {"joke": msg.content})

    # Return the joke to be added to state
    return {"joke": msg.content}


# Build the workflow with memory capabilities
workflow = StateGraph(State)

# Add the memory-aware joke generation node
workflow.add_node("generate_joke", generate_joke)

# Connect the workflow components
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", END)

# Compile with both checkpointing and memory store
chain = workflow.compile(checkpointer=checkpointer, store=memory_store)

# Display the enhanced workflow
save_png(chain.get_graph().draw_mermaid_png(), path="../data/images/chain2.png")
# display(Image(chain.get_graph().draw_mermaid_png()))

# Execute the workflow with thread-based configuration
config = {"configurable": {"thread_id": "1"}}
joke_generator_state = chain.invoke({"topic": "cats"}, config)

# Display the workflow result with rich formatting
console.print("\n[bold cyan]Workflow Result (Thread 1):[/bold cyan]")
pprint(joke_generator_state)

# Retrieve the latest state snapshot from the checkpointer
latest_state = chain.get_state(config)

# Display the complete state snapshot with rich formatting
console.print("\n[bold magenta]Latest Graph State:[/bold magenta]")
pprint(latest_state)

# Execute the workflow with a different thread ID
config = {"configurable": {"thread_id": "2"}}
joke_generator_state = chain.invoke({"topic": "cats"}, config)

# Display the result showing memory persistence across threads
console.print("\n[bold yellow]Workflow Result (Thread 2):[/bold yellow]")
pprint(joke_generator_state)