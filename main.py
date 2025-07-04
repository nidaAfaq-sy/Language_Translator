from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


Translator = Agent(
    name="Language Translator",
    instructions="""
You are a language translator. Your primary task is to translate any Urdu sentence into English.

If someone asks you, "Who are you?" or similar questions about your identity, respond with:
"I am a language translator that translates Urdu into English. I was created by Qaimudin Khuwaja."

Only respond with translations or your identity as instructed. Do not answer unrelated questions.
"""
)


print("Hello! I’m Urdu-to-English Translator 🤖 , (Type 'exit' to quit)")

while True:
    user_input=input("Enter urdu text:")
    if user_input.lower() == "exit":
        print("Exiting the translator bot. Goodbye!")
        break
    
    response = Runner.run_sync(
        Translator,
        input=user_input,
        run_config=config
    )

    print(f"Translated text: {response.final_output}")