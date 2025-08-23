import os
from agents import Agent, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunContextWrapper, TResponseInputItem, function_tool, input_guardrail
from dotenv import load_dotenv
from agents.run import RunConfig
from pydantic import BaseModel
import asyncio

# Load environment variables from .env file.
load_dotenv()

# Fetch Gemini API key.
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

# Configure Gemini via OpenAI compatible API.
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Set up the model (Gemini 2.0 Flash).
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

# Define context with user name and score
class UserInfo(BaseModel):
    name: str
    score: int 

class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str

guardrail_agent = Agent( 
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
)

@input_guardrail
async def math_guardrail( 
    ctx: RunContextWrapper[UserInfo], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context, run_config=config)

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=result.final_output.is_math_homework
    )


# Tool to give feedback based on score
@function_tool
async def give_feedback(wrapper: RunContextWrapper[UserInfo]) -> str:
    """Returns feedback based on the user's score from context."""
    user = wrapper.context
    if user.score >= 80:
        return f"{user.name}, great job, you scored {user.score}!"
    elif user.score >= 50:
        return f"{user.name}, good effort, you scored {user.score}."
    else:
        return f"{user.name}, keep practicing, you scored {user.score}."

# Main function
async def main():
    # Create context
    user_context = UserInfo(name="Alishba", score=95)

    # Define agent
    agent = Agent(
        name="Feedback Agent",
        instructions="You give feedback based on the user's score.",
        model=model,
        tools=[give_feedback],
        input_guardrails=[math_guardrail]
        
    )

    # Get user input
    user_query = input("Enter your query (e.g., 'How did I do?'): ")

    try:
        result = await Runner.run(
        agent,
        input=user_query,
        run_config=config,
        context=user_context
        )
        print(result.final_output)

    except InputGuardrailTripwireTriggered:
        print("Math homework guardrail tripped")

if __name__ == "__main__":
    asyncio.run(main())