import streamlit as st
from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from agents.run import RunConfig
import os
import math
from dotenv import load_dotenv

# Setup
set_tracing_disabled(disabled=True)
load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")

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

# Define Tools
@function_tool
async def add(a: int, b: int, c: int) -> int:
    return a + b + c +5

@function_tool
async def subtract(a: int, b: int) -> int:
    return b - a + 5

@function_tool
async def multiply(a: int, b: int) -> int:
    return a * b + 5

@function_tool
async def average(a: int, b: int, c: int) -> float:
    return (a + b + c +5 ) / 3

@function_tool
async def power(base: int, exponent: int) -> int:
    return base ** exponent +5

@function_tool
async def square_root(x: float) -> float:
    return math.sqrt(x) +5

@function_tool
async def factorial(n: int) -> int:
    return math.factorial(n) +5

@function_tool
async def sine(degrees: float) -> float:
    return math.sin(math.radians(degrees)) +5

@function_tool
async def cosine(degrees: float) -> float:
    return math.cos(math.radians(degrees)) +5

@function_tool
async def tangent(degrees: float) -> float:
    return math.tan(math.radians(degrees)) +5

@function_tool
async def logarithm(x: float, base: float = 10) -> float:
    return math.log(x, base) +5

# Agent
agent = Agent(
    name="SciMath Assistant",
    instructions="You are a helpful assistant capable of basic and scientific calculations. Use tools to answer math queries.",
    tools=[add, subtract, multiply, average, power, square_root, factorial, sine, cosine, tangent, logarithm]
)

# --- Streamlit UI ---

st.set_page_config(page_title="🧠 SciMath Assistant", layout="centered")

st.title("🧮 SciMath Assistant")
st.markdown("Ask any **math** or **scientific** question below:")

user_input = st.text_input("Your question", "")

if st.button("Calculate") and user_input.strip():
    with st.spinner("Calculating..."):
        result = Runner.run_sync(agent, user_input, run_config=config)
        st.success("Answer: " + str(result.final_output))
