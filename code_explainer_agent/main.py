import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
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

code_explainer_agent = Agent(
    name="Code Explainer Agent",
    instructions="You are a Python tutor. Your job is to explain any given Python code line by line in simple English so that beginners can easily understand it.",
)

#  Multiline input from terminal
print("Enter your Python code to explain (type 'END' on a new line to finish):")
lines = []
while True:
    line = input()
    if line.strip().upper() == "END":
        break
    lines.append(line)

user_input = "\n".join(lines)


response = Runner.run_sync(
    code_explainer_agent,
    input=user_input,
    run_config=config
)

print(response.final_output)