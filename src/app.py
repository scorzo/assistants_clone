from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

import openai

import streamlit as st

import json

from dotenv import load_dotenv
import os
import sys



DEBUG_MODE = True
# Embeddings
CHROMA_PATH = "../embeddings.gcp_notes/chroma"


sys.path.append('../')
from newChat.new_chat import NewChat

def debug_print(*args, **kwargs):
    """Print debug messages if DEBUG_MODE is True."""
    if DEBUG_MODE:
        print(*args, **kwargs)

def print_debug_info(header=None):

    debug_print("-" * 80)
    if header:
        debug_print(header)

    debug_print("Model choice:", st.session_state['model_choice'])
    debug_print("Environment variables of interest:")
    if 'OPENAI_API_KEY' in os.environ:
        debug_print("OPENAI_API_KEY:", os.environ['OPENAI_API_KEY'])
    if 'OPENAI_API_BASE' in os.environ:
        debug_print("OPENAI_API_BASE:", os.environ['OPENAI_API_BASE'])
    if 'EMBEDDINGS_OPENAI_API_KEY' in os.environ:
        debug_print("EMBEDDINGS_OPENAI_API_KEY:", os.environ['EMBEDDINGS_OPENAI_API_KEY'])
    if 'OPENAI_BASE_URL' in os.environ:
        debug_print("OPENAI_BASE_URL:", os.environ['OPENAI_BASE_URL'])

    debug_print("-" * 80)

tools = [
    {
        "type": "function",
        "function":
            {
                "name": "get_pizza_info",
                "description": "Get name and price of a pizza of the restaurant",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pizza_name": {
                            "type": "string",
                            "description": "The name of the pizza, e.g. Hawaii",
                        },
                    },
                    "required": ["pizza_name"],
                },
            }},
    {
        "type": "function",
        "function":
            {
                "name": "place_order",
                "description": "Place an order for a pizza from the restaurant",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pizza_name": {
                            "type": "string",
                            "description": "The name of the pizza you want to order, e.g. Margherita",
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "The number of pizzas you want to order",
                            "minimum": 1
                        },
                        "address": {
                            "type": "string",
                            "description": "The address where the pizza should be delivered",
                        },
                    },
                    "required": ["pizza_name", "quantity", "address"],
                },
            }}
]

def place_order(pizza_name, quantity, address):
    if pizza_name not in fake_db["pizzas"]:
        return f"We don't have {pizza_name} pizza!"

    if quantity < 1:
        return "You must order at least one pizza."

    order_id = len(fake_db["orders"]) + 1
    order = {
        "order_id": order_id,
        "pizza_name": pizza_name,
        "quantity": quantity,
        "address": address,
        "total_price": fake_db["pizzas"][pizza_name]["price"] * quantity
    }

    fake_db["orders"].append(order)

    return f"Order placed successfully! Your order ID is {order_id}. Total price is ${order['total_price']}."

def get_pizza_info(pizza_name):
    if pizza_name in fake_db["pizzas"]:
        pizza = fake_db["pizzas"][pizza_name]
        return f"Pizza: {pizza['name']}, Price: ${pizza['price']}"
    else:
        return f"We don't have information about {pizza_name} pizza."

fake_db = {
    "pizzas": {
        "Hawaii": {
            "name": "Hawaii",
            "price": 15.0
        },
        "Margherita": {
            "name": "Margherita",
            "price": 12.0
        }
    },
    "orders": []
}

def vector_search(user_input, conversation_id):

    # used to create embeddings
    llm = ChatOpenAI()

    if conversation_id is None:
        sys.exit("Error: conversation ID is required for this function.")
    chat_history = st.session_state.chat_manager.get_conversation_history(conversation_id)

    # print(json.dumps(chat_history, indent=4))
    # A) Get the value of the env var and save it
    saved_api_base = os.environ.get('OPENAI_API_BASE')
    # B) Temporarily remove the env var during the embeddings call
    os.environ.pop('OPENAI_API_BASE', None)
    print_debug_info("OPENAI_API_BASE temporarily set to None")

    # 1: Build retriever based on vector embeddings
    # Set the OPENAI_API_KEY for OpenAIEmbeddings
    embeddings_openai_api_key = os.getenv('EMBEDDINGS_OPENAI_API_KEY')
    if embeddings_openai_api_key:
        embedding_function = OpenAIEmbeddings(openai_api_key=embeddings_openai_api_key)
        print_debug_info("Using EMBEDDINGS_OPENAI_API_KEY")
    else:
        # default to openai_api_key - should be set
        embedding_function = OpenAIEmbeddings()

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    retriever = db.as_retriever()

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, retriever_prompt)

    # Invoke the retriever chain to get the search query
    search_query_response = retriever_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })

    # C) Make the saved env var value active again
    if saved_api_base is not None:
        os.environ['OPENAI_API_BASE'] = saved_api_base
        print_debug_info("OPENAI_API_BASE set back to original value")

    st.session_state["search_query_response"] = "\n".join([f"Document {i+1}:\nContent: {doc.page_content}\nMetadata: {doc.metadata}\n{'-' * 50}" for i, doc in enumerate(search_query_response)])

    if DEBUG_MODE:
        print("Generated search query response:")
        for i, doc in enumerate(search_query_response):
            print(f"Document {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 50)

    return search_query_response


def get_response(user_input, conversation_id):

    if conversation_id is None:
        sys.exit("Error: conversation ID is required for this function.")

    # 0) do vector search
    vector_docs = vector_search(user_input, conversation_id)
    # Prepare the context from the retrieved documents
    context_string = "\n".join([doc.page_content for doc in vector_docs])

    message_content = request_loop(user_input, conversation_id, context_string)

    return message_content

def request_loop(user_input, conversation_id, context_string, max_iterations=5):

    # - loop over llm calls until it returns a finish reason of "stop" and a response message
    # - forward all responses with finish reason of "tool_calls" to the function calling mechanism
    # - include comprehensive list of all function call responses so far with each llm call

    function_response = []
    iterations = 0

    while iterations < max_iterations:

        response = make_llm_request(user_input, conversation_id, context_string, function_response)
        response_dict = response.dict()
        message = response_dict["choices"][0]["message"]
        finish_reason = response_dict["choices"][0]["finish_reason"]

        if finish_reason == "stop":
            return message["content"]

        if "tool_calls" in message and message["tool_calls"]:
            function_response += function_calls(message)  # Append the result to the list

        iterations += 1

    # Optional: Handle the case where the loop exits due to reaching max_iterations
    # You can return a default value, raise an exception, or handle it in another way
    return "Max iterations reached without stopping condition."


def function_calls(message):
    # Return a single data structure with output from one or more function calls
    function_responses = []
    if "tool_calls" in message and message["tool_calls"]:
        for function_call in message["tool_calls"]:
            function_name = function_call["function"]["name"]
            arguments = json.loads(function_call["function"]["arguments"])

            # Call the locally defined function
            if function_name in globals():
                function = globals()[function_name]
                function_response = function(**arguments)
                print("Function called successfully:", function_response)
                function_responses.append({"name": function_name, "response": function_response})
            else:
                print(f"Function '{function_name}' does not exist in globals.")

        # Return a single data structure with output from one or more function calls
        return function_responses
    else:
        # should not get here
        return [{"name": None, "response": message["content"]}]

def make_llm_request(user_input, conversation_id, context_string, function_responses):

    if conversation_id is None:
        sys.exit("Error: conversation ID is required for this function.")

    chat_history = st.session_state.chat_manager.get_conversation_history(conversation_id)

    # Remove any 'timestamp' fields from the messages
    for message in chat_history:
        message.pop('timestamp', None)

    # Create the messages list, starting with the system context
    messages = [{"role": "system", "content": context_string}]
    # Add the chat history messages to the messages list
    messages.extend(chat_history)

    # user request
    messages.append({"role": "user", "content": user_input})

    # Update to support multiple function call responses if found
    for function_response in function_responses:
        if function_response["name"]:
            messages.append({
                "role": "assistant",
                "function_call": {
                    "name": function_response["name"],
                    "arguments": json.dumps(function_response["response"])
                }
            })
        else:
            messages.append({"role": "assistant", "content": function_response["response"]})

    follow_up_request = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "tools" : tools
    }

    # Print the follow-up request in a human-readable format
    print("-" * 50)
    print("Follow-up Request:")
    print(json.dumps(follow_up_request, indent=4))

    # Make the follow-up request
    response = openai.chat.completions.create(**follow_up_request)

    # Print the response in a human-readable format
    print("Follow-up Response:")
    print(response.model_dump_json(indent=4))
    print("-" * 50)

    return response


# Define the callback function
def on_model_choice_change(header=None):

    if header:
        print_debug_info(header)

    # Clear environment variables
    os.environ.pop('OPENAI_API_KEY', None)
    os.environ.pop('OPENAI_API_BASE', None)
    os.environ.pop('EMBEDDINGS_OPENAI_API_KEY', None)
    os.environ.pop('OPENAI_BASE_URL', None)

    if 'model_choice' not in st.session_state or st.session_state['model_choice'] == "closed source":
        load_dotenv()
        # Print debug information if DEBUG_MODE is True
        print_debug_info("Model choice update completed: closed source")
    else:
        load_dotenv("../runpod.env")
        print_debug_info("Model choice update completed: open source")




# title
user = "Eugene"
st.set_page_config(page_title="MeaningfulAI", page_icon="🤖")
st.title(f"Hello, {user}")

# Sidebar
with st.sidebar:

    # Print the current session state
    print("Session state before:", st.session_state)

    # Set the initial value for 'model_choice' in conversation state
    if 'model_choice' not in st.session_state:
        st.session_state['model_choice'] = "closed source"
        on_model_choice_change('Model choice change initiated')

        print("Session state after:", st.session_state)

    # Create the selectbox with the on_change callback
    st.selectbox(
        "Choose the model source:",
        ["closed source", "open source"],
        index=0 if st.session_state['model_choice'] == "closed source" else 1,
        on_change=on_model_choice_change,
        key="model_choice"
    )


    # Initialize NewChat instance for the user "eugene"
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = NewChat("chat_histories", user)

    st.header("conversations")
    conversations = st.session_state.chat_manager.get_conversations()

    for conversation_id in conversations:
        conversation_history = st.session_state.chat_manager.get_conversation_history(conversation_id)
        if conversation_history:  # Check if the conversation history is not empty
            first_message_content = conversation_history[0]["content"]  # Get the first message content of the conversation
            button_label = " ".join(first_message_content.split()[:5])  # Use the first 5 words as the button label
            # creates a button and runs "if" stmt if button pressed
            if st.button(button_label, key=conversation_id):
                # clears and sets the conversation history to the button clicked
                st.session_state.chat_manager.set_active_conversation(conversation_id)
                # clear the search query response text area
                st.session_state["search_query_response"] = ""


# input field
user_query = st.chat_input("Type your message here...")

# handle input field state changes
if user_query is not None and user_query != "":

    # must have a conversation_id to call this
    response = get_response(user_query, st.session_state.chat_manager.get_active_conversation())
    st.session_state.chat_manager.append_message(st.session_state.chat_manager.get_active_conversation(), HumanMessage(content=user_query))
    st.session_state.chat_manager.append_message(st.session_state.chat_manager.get_active_conversation(), AIMessage(content=response))




# keep polling for chat history and update if it changes
chat_history = st.session_state.chat_manager.get_conversation_history(st.session_state.chat_manager.get_active_conversation())

for message in chat_history:
    if message["role"] == "assistant":
        with st.chat_message("AI"):
            st.write(message["content"])
    elif message["role"] == "user":
        with st.chat_message("Human"):
            st.write(message["content"])

if 'search_query_response' not in st.session_state:
    st.session_state["search_query_response"] = ""

if 'show_response' not in st.session_state:
    st.session_state['show_response'] = False

# Add a checkbox to toggle the visibility of the search query response
show_response = st.checkbox('Show Generated Search Query Response', value=st.session_state['show_response'])

if show_response:
    # Display the search query response in a scrolling textarea
    st.text_area("Generated Search Query Response:", value=st.session_state["search_query_response"], height=300, key="search_query_response_area")



