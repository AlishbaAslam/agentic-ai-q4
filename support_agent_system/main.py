import os
from dotenv import load_dotenv
from typing import Literal
from agents import Agent, ItemHelpers, ModelSettings, Runner, AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel, RunContextWrapper, function_tool, output_guardrail, GuardrailFunctionOutput
import asyncio
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

# Set up the model (Gemini-1.5-Flash-Latest).
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash-latest",
    openai_client=external_client
)

# Disable tracing for simplicity.
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

class UserInfo(BaseModel):
    name: str
    is_premium_user: bool = False
    issue_type: Literal["billing", "technical", "general"] = "general"

class ApologyOutput(BaseModel):
    has_apology: bool

# Guardrail Agent
guardrail_agent = Agent(
    name="Apology Guardrail",
    instructions=(
        "Check if the output contains apology words like 'sorry', 'apologies', or 'regret'. "
        "Set has_apology=True if detected."
    ),
    output_type=ApologyOutput,
)

@output_guardrail
async def no_apologies_guardrail(wrapper: RunContextWrapper[None], agent: Agent, output: str) -> GuardrailFunctionOutput:
    
    result = await Runner.run(guardrail_agent, output, context=wrapper.context, run_config=config)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.has_apology,
    )

def is_premium(ctx: RunContextWrapper[UserInfo], agent: Agent) -> bool:
    return ctx.context.is_premium_user

@function_tool(is_enabled=is_premium)
def issue_refund(amount: int, reason: str):
    """Issues a refund to a premium user."""
    return f"Refund of ${amount} for '{reason}' has been processed."

def is_technical(ctx: RunContextWrapper[UserInfo], agent: Agent) -> bool:
    return ctx.context.issue_type == "technical"

@function_tool(is_enabled=is_technical)
def restart_service(service_name: str):
    """Restarts a technical service."""
    return f"The '{service_name}' service has been restarted."

# Specialized Agents
billing_agent = Agent[UserInfo](
    name="Billing Agent",
    instructions="You are a billing specialist. You can issue refunds to premium users.",
    tools=[issue_refund],
    model_settings=ModelSettings(tool_choice="required")
)

technical_agent = Agent[UserInfo](
    name="Technical Agent",
    instructions="You are a technical support specialist. You can restart services.",
    tools=[restart_service],
    model_settings=ModelSettings(tool_choice="required")
)

# Define the Triage Agent with Handoffs
triage_agent = Agent[UserInfo](
    name="Triage Agent",
    instructions="You are a triage agent. Your job is to determine the user's issue and hand them off to the correct specialist.",
    handoffs=[billing_agent, technical_agent],
    output_guardrails=[no_apologies_guardrail] 
)

async def main():
    user_context = UserInfo(name="Alishba", is_premium_user=True, issue_type="billing")
    print("Welcome to the Console-Based Support Agent System. How can I help you today?")

    while True:
        user_input = input("> ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Simple keyword routing
        if "refund" in user_input.lower():
            user_context.issue_type = "billing"
        elif "restart" in user_input.lower():
            user_context.issue_type = "technical"
        else:
            user_context.issue_type = "general"

        result = Runner.run_streamed(triage_agent, user_input, context=user_context, run_config=config)

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                continue

            elif event.type == "agent_updated_stream_event":
                # Sirf Triage → Specialist handoff print karo
                if event.new_agent.name != "Triage Agent":
                    print(f"[Handoff] Switching from Triage Agent → {event.new_agent.name}")
                continue

            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    continue
                    
                elif event.item.type == "tool_call_output_item":
                    print(f"[Tool Output] {event.item.output}")
                    
                elif event.item.type == "message_output_item":
                    print(f"[Response]\n{ItemHelpers.text_message_output(event.item)}")
                else:
                    pass

if __name__ == "__main__":
    asyncio.run(main())