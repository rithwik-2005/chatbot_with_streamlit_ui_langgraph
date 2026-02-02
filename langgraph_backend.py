from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

#model defining 
model=ChatOpenAI(model="gpt-4o-mini")

#state defining
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]    #for chat-models we need to add conversations 

#python function for node creation
def chat_node(state:ChatState):
    messages=state["messages"]
    response=model.invoke(messages)
    return {"messages": [response]}

#initalisation checkpointer
checkpointer=InMemorySaver()

#creating nodes
graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
#creating edges
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot=graph.compile(checkpointer=checkpointer)

