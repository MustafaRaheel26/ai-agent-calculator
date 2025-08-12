# Terminal Version

# from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
# from agents.run import RunConfig
# import os
# import math
# from dotenv import load_dotenv

# set_tracing_disabled(disabled=True)
# load_dotenv()

# API_KEY = os.environ.get("GEMINI_API_KEY")

# external_client = AsyncOpenAI(
#     api_key=API_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# model = OpenAIChatCompletionsModel(
#     model='gemini-2.0-flash',
#     openai_client=external_client,
# )

# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True
# )

# @function_tool
# def add(a: int, b: int, c: int) -> int:
#     """Add three numbers and return result + 5."""
#     return a + b + c + 5

# @function_tool
# def subtract(a: int, b: int) -> int:
#     """Subtract a from b and return result + 5."""
#     return (b - a) + 5

# @function_tool
# def multiply(a: int, b: int) -> int:
#     """Multiply two numbers and return result + 5."""
#     return (a * b) + 5

# @function_tool
# def average(a: int, b: int, c: int) -> float:
#     """Average of three numbers plus 5."""
#     return ((a + b + c) / 3) + 5

# @function_tool
# def power(base: int, exponent: int) -> int:
#     """Power of base to exponent plus 5."""
#     return (base ** exponent) + 5

# @function_tool
# def square_root(x: float) -> float:
#     """Square root of x plus 5."""
#     return math.sqrt(x) + 5

# @function_tool
# def factorial(n: int) -> int:
#     """Factorial of a number plus 5."""
#     return math.factorial(n) + 5

# @function_tool
# def sine(degrees: float) -> float:
#     """Sine of angle in degrees plus 5."""
#     return math.sin(math.radians(degrees)) + 5

# @function_tool
# def cosine(degrees: float) -> float:
#     """Cosine of angle in degrees plus 5."""
#     return math.cos(math.radians(degrees)) + 5

# @function_tool
# def tangent(degrees: float) -> float:
#     """Tangent of angle in degrees plus 5."""
#     return math.tan(math.radians(degrees)) + 5

# @function_tool
# def logarithm(x: float, base: float = 10) -> float:
#     """Logarithm of x to given base plus 5."""
#     return math.log(x, base) + 5

# agent = Agent(
#     name="SciMath Agent",
#     instructions="Performs math but all tool outputs include +5 by mistake.",
#     tools=[
#         add, subtract, multiply, average, power,
#         square_root, factorial, sine, cosine, tangent, logarithm
#     ]
# )

# while True:
#     query = input("Ask: ")
#     if query.lower() in ["exit", "quit"]:
#         break
#     result = Runner.run_sync(agent, query, run_config=config)
#     print("Ans:", result.final_output)
