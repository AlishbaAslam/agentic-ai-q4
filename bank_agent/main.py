import os
from dotenv import load_dotenv
from agents import Agent, GuardrailFunctionOutput, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, input_guardrail
from agents.run import RunContextWrapper
from agents.run import RunConfig
from pydantic import BaseModel

# Load environment variables from .env file.
load_dotenv()

# Fetch Gemini API Key.
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

# Configure Gemini via OpenAI-compatible API.
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Set up the model (Gemini-2.0-Flash).
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Disable tracing for simplicity.
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

class Account(BaseModel):
    name: str
    pin: int

class Guardrail_output(BaseModel):
    is_bank_related: bool

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="check if the user is asking you bank related queries",
    output_type=Guardrail_output
)

@input_guardrail
async def check_bank_related(ctx: RunContextWrapper[None], agent: Agent, input: str) -> GuardrailFunctionOutput:

    result = await Runner.run(guardrail_agent, input, context=ctx.context, run_config=config)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_bank_related
    )

def check_user(ctx: RunContextWrapper[Account], agent: Agent) -> bool:
    if ctx.context.name == "Alishba" and ctx.context.pin == 1234:
        return True
    else:
        print("User authentication failed.")
        return False

@function_tool(is_enabled=check_user)
def check_balance(account_number: str) -> str:
    print("User is authenticated.")
    print(f"Checking balance for account number: {account_number}")
    return f"The balance of account {account_number} number is $1000000"

bank_agent = Agent(
    name="Bank Agent",
    instructions="You are a bank agent. You help customers with their questions.",
    tools=[check_balance],
    input_guardrails=[check_bank_related]
)

user_context = Account(name="Alishba", pin=1234)

result = Runner.run_sync(
    bank_agent,
    "I want to check my balance. My account number is 309473804",
    context=user_context,
    run_config=config
)

print(result.final_output)