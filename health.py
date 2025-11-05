import os
from typing import TypedDict , Annotated , List
from langgraph.graph import StateGraph , END
from langchain_core.messages import HumanMessage , AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image

API_KEY = "<API_KEY>"
class Plannerstate(TypedDict):
    messages : Annotated[List[HumanMessage | AIMessage], "This is the message"]
    product: str
    concerns: list[str]
    health_suitability_profile: str

from langchain_groq import ChatGroq
llm = ChatGroq(
    temperature=0,
    groq_api_key = "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",
    model_name = "llama-3.3-70b-versatile")
health_suitability_profile_prompt = ChatPromptTemplate.from_messages(
    [("system","you are a helpful health assistant.create a health_suitability_profile  for {product} based on user's concerns:{concerns}. provide a brief bulleted health_suitability_profile"),("human","create a health_suitability_profile for product")]
) 
def input_product(state:Plannerstate) -> Plannerstate:
    print(f"please enter the product you want to generate health_suitability_profile")
    user_message = input("Your Input: ")
    return{
    **state,
    "product": user_message,
    "messages": state['messages']+ [HumanMessage(content=user_message)]
}
def input_concerns(state:Plannerstate) -> Plannerstate:
    print(f"please enter your concerns : {state['product']} (comma-seperated)")
    user_message = input("Your Input: ")
    return{
    **state,
    "concerns": [concerns.strip() for concerns in user_message.split(",")],
    "messages": state['messages']+ [HumanMessage(content=user_message)]
}
def create_health_suitability_profile(state:Plannerstate) -> Plannerstate:
    print(f"create an health_suitability_profile for {state['product']} based on concerns: {', '.join(state['concerns'])}")
    response = llm.invoke(health_suitability_profile_prompt.format_messages(
    product=state['product'], concerns=', '.join(state['concerns'])
))

    print("\nfinal health_suitability_profile :")
    print(response.content)
    return{
    **state, 
    "messages": state['messages']+ [AIMessage(content=response.content)],"health_suitability_profile": response.content}

workflow = StateGraph(Plannerstate)
workflow.add_node("input_product",input_product)
workflow.add_node("input_concerns",input_concerns)
workflow.add_node("create_health_suitability_profile",create_health_suitability_profile)
workflow.set_entry_point("input_product")
workflow.add_edge("input_product","input_concerns")
workflow.add_edge("input_concerns","create_health_suitability_profile")
workflow.add_edge("create_health_suitability_profile",END)
app = workflow.compile()

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method = MermaidDrawMethod.API
        )
    )
)

def travel_planner(user_request:str):
        print(f"Initial Request :{user_request}\n")
        state = {
            "messages" : [HumanMessage(content=user_request)],
            "product":"",
            "concerns": [],
            "health_suitability_profile":"",
        }
        for output in app.stream(state):
            pass
user_request = "I want to generate a health_suitability_profile for given product"
travel_planner(user_request)

import gradio as gr
from typing import TypedDict , Annotated , List
from langgraph.graph import StateGraph , END
from langchain_core.messages import HumanMessage , AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod


class Plannerstate(TypedDict):
    messages : Annotated[List[HumanMessage | AIMessage], "This is the message"]
    product: str
    concerns: list[str]
    health_suitability_profile: str
from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0,
    groq_api_key = "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",
    model_name = "llama-3.3-70b-versatile")
health_suitability_profile_prompt = ChatPromptTemplate.from_messages(
    [("system","you are a helpful health assistant.create a health_suitability_profile  for {product} based on user's concerns:{concerns}. provide a brief bulleted health_suitability_profile"),("human","create a health_suitability_profile for product")]
) 
def input_product(product:str ,state:Plannerstate) -> Plannerstate:
    
    return{
    **state,
    "product": product,
    "messages": state['messages']+ [HumanMessage(content=product)]
}
def input_concerns(concerns : str, state:Plannerstate) -> Plannerstate:
    
    return{
    **state,
    "concerns": [concerns.strip() for concerns in concerns.split(",")],
    "messages": state['messages']+ [HumanMessage(content=concerns)]
}
def create_health_suitability_profile(state: Plannerstate) -> str:
    response = llm.invoke(health_suitability_profile_prompt.format_messages(
        product=state["product"],
        concerns=", ".join(state["concerns"])
    ))

    # 🛠 Fix: If response is a tuple, extract the first item
    if isinstance(response, tuple):
        response = response[0]

    state["health_suitability_profile"] = response.content
    state["messages"] += [AIMessage(content=response.content)]
    return response.content


def health_planner(product: str, concerns: str):
    state = {
          "messages": [],
          "product": "",
          "concerns": [],
          "health_suitability_profile": "",
           }
    
    state = input_product(product,state)
    state = input_concerns(concerns,state)
    health_suitability_profile = create_health_suitability_profile(state)
    return health_suitability_profile

interface = gr.Interface(
    fn = health_planner,
    theme='gr.themes.Soft()',
    inputs=[
         gr.Textbox(label="Enter the product for your health suitability profile "),
         gr.Textbox(label="Enter your concerns (comma-separated)"),
    ],
    outputs=gr.Textbox(label="Generated health suitability profile"),
    title="FitSense -Health Suitability Profile Generator",
    description="Enter a product and your concerns to generate a health suitability profile"
)

interface.launch(share=True)