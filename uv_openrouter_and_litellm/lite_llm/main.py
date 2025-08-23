import os
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from agents.run import RunConfig

# Load environment variables from .env file.
load_dotenv()

# Fetch Gemini API key.
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

# Set up the model (Gemini 2.0 Flash).
model = LitellmModel(
    model="gemini/gemini-2.0-flash",
    api_key=gemini_api_key
)

# Disable tracing for simplicity.
config = RunConfig(
    model=model,
    tracing_disabled=True
)

agent = Agent(
    name="Assistant",
    instructions="You are a Helpful Assistant.",
    model=model
)

result = Runner.run_sync(
starting_agent=agent,
input="Hi! Who are you?",
run_config=config
)
print(result.final_output)