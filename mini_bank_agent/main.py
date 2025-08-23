from agents import( 
Agent,
GuardrailFunctionOutput,
InputGuardrailTripwireTriggered,
OutputGuardrailTripwireTriggered,
Runner,
AsyncOpenAI,
OpenAIChatCompletionsModel,
function_tool,
input_guardrail,
output_guardrail,
)
import os
from pydantic import BaseModel
from agents.run import RunContextWrapper, RunConfig
from dotenv import load_dotenv
import asyncio

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

class GuardrailOutput(BaseModel):
    is_bank_related: bool
    reasoning: str

input_guardrail_agent = Agent(
    name="Input Guardrail Agent",
    instructions="check if the user is asking you bank related queries. Return a structured output with 'is_bank_related' as a boolean and 'reasoning' explaining your decision.",
    output_type=GuardrailOutput,
)

output_guardrail_agent = Agent(
    name="Output Guardrail Agent",
    instructions="Check if the output includes any bank-related content. Return a structured output with 'is_bank_related' as a boolean and 'reasoning' explaining your decision.",
    output_type=GuardrailOutput,
)

@input_guardrail
async def check_bank_related_input(ctx: RunContextWrapper[None], agent: Agent, input: str) -> GuardrailFunctionOutput:

    result = await Runner.run(input_guardrail_agent, input, context=ctx.context, run_config=config)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_bank_related,
    )

@output_guardrail
async def check_bank_related_output(ctx: RunContextWrapper[None], agent: Agent, output: str)->GuardrailFunctionOutput:
    
    result = await Runner.run(output_guardrail_agent, output, context=ctx.context, run_config = config)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_bank_related,
    )

def check_user(ctx: RunContextWrapper[Account], agent: Agent) -> bool:
    if ctx.context.name == "Alishba" and ctx.context.pin == 1234:
        return True
    else:
        print("⚠️ Access denied. Incorrect credentials.")
        return False

@function_tool(is_enabled=check_user)
def check_balance(account_number: str) -> str:
    accounts = {
        "309473804": "$1,000,000",
        "123456789": "$5,000",
        "987654321": "$20,000"
    }
    balance = accounts.get(account_number)
    if balance:
        return f"The balance of account {account_number} is {balance}"
    return "Account not found."
    
    
greeting_agent = Agent(
    name="Greeting Agent",
    instructions="You greet the customer with their names and ask how you can assist them today."
)

support_agent = Agent(
    name="Support Agent",
    instructions=("You are a support agent. You handle general queries and complaints from customers. You also provide answers to frequently asked questions (FAQs) related to banking."
    )
)

bank_agent = Agent(
    name="Bank Agent",
    instructions="You are a bank agent assisting customers with banking needs. You answer questions, provide guidance, and hand off conversations to specialized agents like Greeting Agent and Support Agent when appropriate. You may use tools such as check_balance to assist customers quickly. Always ensure inputs follow bank policies and validation rules.",
    handoffs=[greeting_agent, support_agent],
    tools=[check_balance],
    input_guardrails=[check_bank_related_input],
    output_guardrails=[check_bank_related_output]
)

async def main():
    user_context = Account(name="Alishba", pin=1234)
    user_input = input("✅ Welcome! Authentication successful. \n🔐 PIN verified. You're now securely logged in. \nHow can I assist you with your banking needs today? \nPlease enter your request: ")

    try:
        result = await Runner.run(
            starting_agent=bank_agent,
            input=user_input,
            context=user_context,
            run_config=config
        )
        print(result.final_output)
        
    except InputGuardrailTripwireTriggered:
        print("Input guardrail tripped!")
    except OutputGuardrailTripwireTriggered:
        print("Output guardrail tripped!")

if __name__ == "__main__":
    asyncio.run(main())