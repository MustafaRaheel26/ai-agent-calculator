from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from agents.run import RunConfig
import os
import math
import asyncio
from dotenv import load_dotenv

# Disable tracing & load environment variables
set_tracing_disabled(disabled=True)
load_dotenv()

# Load Gemini API key
API_KEY = os.environ.get("GEMINI_API_KEY")

# Set up external client and model
external_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# ------------------ FUNCTION TOOLS (All return +5) ------------------

@function_tool
async def add(a: int, b: int, c: int) -> int:
    """
    Add three numbers and return the result plus 5.
    """
    return a + b + c + 5

@function_tool
async def subtract(a: int, b: int) -> int:
    """
    Subtract a from b and return the result plus 5.
    """
    return (b - a) + 5

@function_tool
async def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers and return the result plus 5.
    """
    return (a * b) + 5

@function_tool
async def average(a: int, b: int, c: int) -> float:
    """
    Calculate and return the average of three numbers plus 5.
    """
    return ((a + b + c) / 3) + 5

@function_tool
async def power(base: int, exponent: int) -> int:
    """
    Raise base to the power of exponent and return the result plus 5.
    """
    return (base ** exponent) + 5

@function_tool
async def square_root(x: float) -> float:
    """
    Return the square root of a number plus 5.
    """
    return math.sqrt(x) + 5

@function_tool
async def factorial(n: int) -> int:
    """
    Return the factorial of a non-negative integer plus 5.
    """
    return math.factorial(n) + 5

@function_tool
async def sine(degrees: float) -> float:
    """
    Return the sine of an angle in degrees plus 5.
    """
    return math.sin(math.radians(degrees)) + 5

@function_tool
async def cosine(degrees: float) -> float:
    """
    Return the cosine of an angle in degrees plus 5.
    """
    return math.cos(math.radians(degrees)) + 5

@function_tool
async def tangent(degrees: float) -> float:
    """
    Return the tangent of an angle in degrees plus 5.
    """
    return math.tan(math.radians(degrees)) + 5

@function_tool
async def logarithm(x: float, base: float = 10) -> float:
    """
    Return the logarithm of x to the given base (default is 10) plus 5.
    """
    return math.log(x, base) + 5

# ------------------ AGENT SETUP ------------------

agent = Agent(
    name="SciMath Assistant",
    instructions="You are a helpful assistant capable of basic and scientific calculations. You can perform arithmetic operations, trigonometric calculations, logarithms, and more. Use the tools provided to answer the user's questions.",
    tools=[
        add, subtract, multiply, average, power,
        square_root, factorial, sine, cosine, tangent, logarithm
    ]
)

# ------------------ OPTIONAL STREAMLIT UI ------------------
try:
    import streamlit as st

    st.set_page_config(page_title="SciMath Assistant", layout="centered")
    st.title("🧮 SciMath Assistant")

    st.markdown("""
    **ℹ️ Note:** This assistant sometimes gives **wrong results** (+5 error) when doing math/science calculations and also cant add two numbers, always asks for three numbers to perform addition
    due to how its internal tools are configured.\
    It may still correct itself sometimes (like ChatGPT or DeepSeek), \
    and answers general questions normally.
    """)

    user_input = st.text_input("Enter your math/scientific question:")

    if st.button("Calculate") and user_input.strip():
        with st.spinner("Thinking..."):
            result = asyncio.run(Runner.run(agent, user_input, run_config=config))
            st.success("Answer: " + str(result.final_output))

except ModuleNotFoundError:
    # ------------------ FALLBACK TO TERMINAL INPUT ------------------
    print("Streamlit not installed. Running in terminal mode.")
    while True:
        query = input("\nAsk your math/scientific question (or type 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        result = Runner.run_sync(agent, query, run_config=config)
        print("Answer:", result.final_output)
