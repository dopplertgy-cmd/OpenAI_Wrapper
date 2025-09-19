import os
import openai
from dotenv import load_dotenv


def load_api_key():
    """
    Load the OpenAI API key from an environment variable or .env file.
    """
    # Load from .env file if present
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return api_key


def get_gpt5_response(prompt: str, system_message: str = None) -> str:
    """
    Send a prompt to the OpenAI GPT-5 model and return the response.

    :param prompt: The user prompt to send to the model.
    :param system_message: Optional system message to set the behavior of the assistant.
    :return: The assistant's reply as a string.
    """
    api_key = load_api_key()
    openai.api_key = api_key

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-5",
        messages=messages
    )
    # Extract the content of the assistant's reply
    return response["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    print("Welcome to the GPT-5 interactive prompt. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        try:
            reply = get_gpt5_response(user_input)
            print(f"GPT-5: {reply}")
        except Exception as e:
            print(f"Error: {e}")

