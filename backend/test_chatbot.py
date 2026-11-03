
from app.chatbot import chat_with_ai


print("=" * 50)
print("Terminal AI Chatbot")
print("Type 'exit' to quit")
print("=" * 50)


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":

        print("\nGoodbye!")
        break

    if not user_input.strip():
        print("\nPlease enter a message.\n")
        continue

    response = chat_with_ai(user_input)

    print("\nAI:", response)




