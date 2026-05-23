import cohere
from rich import print
from dotenv import dotenv_values, load_dotenv

# Load environment variables
load_dotenv()

# Read .env file
env_vars = dotenv_values(".env")

# Get API key
CohereAPIKey = env_vars.get("CohereAPIKey")

# Initialize Cohere client
co = cohere.Client(api_key=CohereAPIKey)

# Supported function labels
funcs = [
    "exit",
    "general",
    "realtime",
    "open",
    "close",
    "play",
    "generate image",
    "system",
    "content",
    "google search",
    "youtube search",
    "reminder"
]

# Message storage
messages = []

# Preamble
preamble = """
You are an advanced Decision-Making Model.

Your task is ONLY to classify user queries.

Do NOT answer questions.

You must classify the query into one of the predefined labels.

Available labels:

general
realtime
open
close
play
generate image
system
content
google search
youtube search
reminder
exit

Rules:

- Use 'general' for conversational or educational queries.
- Use 'realtime' for current/up-to-date information.
- Use 'open' for opening apps/websites.
- Use 'close' for closing apps.
- Use 'play' for music/video playback.
- Use 'generate image' for image generation requests.
- Use 'system' for volume, brightness, shutdown, restart, etc.
- Use 'content' for writing content/code/emails/articles.
- Use 'google search' for searching on Google.
- Use 'youtube search' for YouTube searches.
- Use 'reminder' for reminders/tasks.
- Use 'exit' when user wants to quit conversation.

Examples:

User: who is narendra modi
Assistant: general who is narendra modi

User: what is today's weather
Assistant: realtime what is today's weather

User: open chrome
Assistant: open chrome

User: close spotify
Assistant: close spotify

User: play let her go
Assistant: play let her go

User: mute volume
Assistant: system mute volume

User: increase brightness
Assistant: system increase brightness

User: search ai tools on google
Assistant: google search ai tools

User: search lofi songs on youtube
Assistant: youtube search lofi songs

User: remind me to study at 8pm
Assistant: reminder study at 8pm

User: bye
Assistant: exit

IMPORTANT:
- ONLY return labels and query.
- NEVER explain anything.
- NEVER answer the query.
- Multiple tasks should be comma separated.
"""

# Chat examples
ChatHistory = [
    {"role": "User", "message": "how are you"},
    {"role": "Chatbot", "message": "general how are you"},

    {"role": "User", "message": "open chrome"},
    {"role": "Chatbot", "message": "open chrome"},

    {"role": "User", "message": "play believer"},
    {"role": "Chatbot", "message": "play believer"},

    {"role": "User", "message": "increase brightness"},
    {"role": "Chatbot", "message": "system increase brightness"},

    {"role": "User", "message": "what is weather today"},
    {"role": "Chatbot", "message": "realtime what is weather today"},
]


# Main Function
def FirstLayerDMM(prompt: str = "test"):

    try:

        # Create response stream
        stream = co.chat_stream(
            model="command-a-03-2025",
            message=prompt,
            temperature=0.1,
            chat_history=ChatHistory,
            prompt_truncation="OFF",
            connectors=[],
            preamble=preamble
        )

        # Store generated text
        response = ""

        # Read stream
        for event in stream:
            if event.event_type == "text-generation":
                response += event.text

        # Clean response
        response = response.replace("\n", "")
        response = response.strip()

        # Split multiple tasks
        tasks = response.split(",")

        # Final filtered tasks
        filtered_tasks = []

        for task in tasks:

            task = task.strip().lower()

            for func in funcs:

                if task.startswith(func):
                    filtered_tasks.append(task)
                    break

        # Fallback
        if not filtered_tasks:
            filtered_tasks.append(f"general {prompt}")

        return filtered_tasks

    except Exception as e:

        return [f"Error : {e}"]


# Main loop
if __name__ == "__main__":

    while True:

        user_input = input(">>> ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        result = FirstLayerDMM(user_input)

        print(result)