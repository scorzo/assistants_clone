import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage

class NewChat:
    def __init__(self, parent_folder, username):
        self.parent_folder = Path(parent_folder)
        self.username = username
        self.user_folder = self.parent_folder / self.username
        self.user_folder.mkdir(parents=True, exist_ok=True)
        self.active_conversation = None  # Initialize active_session attribute

    def set_active_conversation(self, conversation_id):
        """Sets the active conversation ID."""
        self.active_conversation = conversation_id

    def get_active_conversation(self):
        """Returns the active session ID or creates a new one if not found."""
        if self.active_conversation is None:
            self.active_conversation = self.create_new_conversation()
        return self.active_conversation

    def get_or_create_active_conversation(self):
        """Returns the active conversation ID or creates a new one if not found."""
        if self.active_conversation is None:
            self.active_conversation = self.create_new_conversation()
        return self.active_conversation

    def get_conversation_file(self, conversation_id):
        return self.user_folder / f"{conversation_id}.json"

    def append_message(self, conversation_id, message):
        conversation_file = self.get_conversation_file(conversation_id)
        chat_history = []  # Initialize chat_history as an empty list
        if conversation_file.exists():
            with open(conversation_file, "r") as file:
                chat_history = json.load(file)  # Update chat_history with existing conversation
        # Construct the dictionary representation of the message
        if isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content, "timestamp": datetime.now().isoformat()}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content, "timestamp": datetime.now().isoformat()}
        else:
            raise TypeError("Unsupported message type")
        chat_history.append(message_dict)
        with open(conversation_file, "w") as file:
            json.dump(chat_history, file, indent=2)



    def get_conversation_history(self, conversation_id):
        conversation_file = self.get_conversation_file(conversation_id)
        if conversation_file.exists():
            with open(conversation_file, "r") as file:
                return json.load(file)
        return []

    def create_new_conversation(self):
        conversation_id = str(uuid.uuid4())
        self.get_conversation_file(conversation_id)  # Ensure the file and directories are created
        return conversation_id

    def get_conversations(self):
        return [f.stem for f in self.user_folder.glob("*.json")]

    def print_conversation_tree(self, directory=None, indent=""):
        if directory is None:
            directory = self.parent_folder
            print(directory)

        for item in directory.iterdir():
            if item.is_dir():
                print(f"{indent}├── {item.name}")
                self.print_conversation_tree(item, indent + "│   ")
            else:
                print(f"{indent}└── {item.name}")

    def print_conversation(self, conversation_id):
        conversation_history = self.get_conversation_history(conversation_id)
        print(json.dumps(conversation_history, indent=2))

# Usage example
if __name__ == "__main__":
    chat_manager = NewChat("chat_histories", "eugene")
    new_conversation_id = chat_manager.create_new_conversation()
    print(f"New conversation created with ID: {new_conversation_id}")

    chat_manager.append_message(new_conversation_id, HumanMessage(content="my name is eugene. what is your name?"))
    chat_manager.append_message(new_conversation_id, AIMessage(content="Hello Eugene, I'm Bob. How can I assist you today?"))

    print("Conversation in the new conversation:")
    chat_manager.print_conversation(new_conversation_id)

    another_conversation_id = chat_manager.create_new_conversation()
    chat_manager.append_message(another_conversation_id, HumanMessage(content="What's the weather like today?"))
    chat_manager.append_message(another_conversation_id, AIMessage(content="It's sunny and warm outside."))

    print("Conversation in the second conversation:")
    chat_manager.print_conversation(another_conversation_id)

    print("Conversation history for the first conversation:")
    history = chat_manager.get_conversation_history(new_conversation_id)
    print(history)

    print("List of all conversation IDs for user 'eugene':")
    conversations = chat_manager.get_conversations()
    print(conversations)

    print("Conversation tree structure:")
    chat_manager.print_conversation_tree()
