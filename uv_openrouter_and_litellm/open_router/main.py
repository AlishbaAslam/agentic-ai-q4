import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv
from agents.run import RunConfig

# Load environment variables from .env file.
load_dotenv()

# Fetch Openrouter API key.
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file!")

# Configure Openrouter via OpenAI compatible API.
external_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Set up the model.
model = OpenAIChatCompletionsModel(
    model="deepseek/deepseek-r1-0528:free",
    openai_client=external_client
)

# Disable tracing for simplicity.
config = RunConfig(
    model=model,
    model_provider=external_client,
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

