import os
from dotenv import load_dotenv
from agents import Agent, GuardrailFunctionOutput, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, input_guardrail, InputGuardrailTripwireTriggered, ModelSettings
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

class User(BaseModel):
    name: str
    member_id: int

class GuardrailOutput(BaseModel):
    is_library_related: bool

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="Return true if the query is about library (books, availability, timings). Otherwise false.",
    output_type=GuardrailOutput,
)

@input_guardrail
async def check_library_related(ctx: RunContextWrapper[None], agent: Agent, input: str) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context, run_config=config)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_library_related
    )

def is_valid_member(ctx: RunContextWrapper[User], agent: Agent) -> bool:
    return ctx.context.member_id in [1001, 1002, 1003]  # valid members list

book_db = {
    "Atomic Habits": 5,
    "The Great Gatsby": 2,
    "AI Revolution": 0,
}

@function_tool
def search_book(book_name: str) -> str:
    """
    Use this tool when user asks 'Do you have ...' or 'Is ... available?'
    """
    if book_name in book_db:
        return f"Yes, '{book_name}' is available in the library."
    else:
        return f"'{book_name}' is not available in the library."

@function_tool(is_enabled=is_valid_member)
def check_availability(book_name: str) -> str:
    """
    Use this tool when user asks 'How many copies' or 'Check availability'.
    """
    if book_name in book_db:
        copies = book_db[book_name]
        return f"There are {copies} copies of '{book_name}' available."
    else:
        return f"'{book_name}' is not found in the library records."

@function_tool
def library_timings() -> str:
    """
    Returns the library opening and closing time
    """
    return "The library is open from 9 AM to 6 PM, Monday to Saturday."

def dynamic_instruction(ctx: RunContextWrapper[User], agent: Agent) -> str:
    return f"""
    You are a helpful library assistant. Address the user by their name: {ctx.context.name}.

    - If the user asks 'Do you have ...' or 'Is ... available?', use the search_book tool.
    - If the user asks 'How many copies...' or 'Check availability', use the check_availability tool.
    - If the user asks about library hours, use the library_timings tool.
    - If the user asks a query that combines checking if a book exists and its availability (e.g., 'Do you have ... and how many copies?'), use both search_book and check_availability tools and combine their outputs.
    """

library_agent = Agent[User](
    name="Library Agent",
    instructions=dynamic_instruction,
    tools=[search_book, check_availability, library_timings],
    input_guardrails=[check_library_related],
    model_settings=ModelSettings(
        temperature=0.2,
        tool_choice='auto', 
        parallel_tool_calls=None,
    ),
    tool_use_behavior='run_llm_again'
)

user_context = User(name="Alishba", member_id=1001)

queries = [
    "Do you have Atomic Habits and how many copies are available?",
    "Is The Great Gatsby available and what are the library hours?",
    "Tell me about Python programming.",  # Non-library query
]

for q in queries:
    print("\n--- User Query:", q)
    try:
        result = Runner.run_sync(library_agent, q, context=user_context, run_config=config)
        print("Assistant:", result.final_output)
    except InputGuardrailTripwireTriggered:
        print(" Guardrail triggered! The query is not related to library services.")