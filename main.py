#!/usr/bin/env python3
"""
Property Insights Assistant v3 - Streamlit Web App
- LLM (Gemini/OpenAI) handles NL and decides when to call tools
- Tools query the CSV dataset (authoritative facts)
- Short-term in-memory session memory with inactivity timeout (default 10 minutes)
"""

import os
import re
import time
import pandas as pd
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import streamlit as st
import asyncio

# Agent SDK imports (from your sample)
from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from agents.run import RunConfig

# -------------------- CONFIG --------------------
DATA_PATH = "OLIdUj.csv"
MEMORY_TURNS = 10                 # keep last N turns (user + assistant counted separately)
INACTIVITY_TIMEOUT = 600          # seconds after which the session memory is cleared (600s = 10 min)
PRICE_SCALE = {"k": 1_000, "m": 1_000_000, "mn": 1_000_000}

# -------------------- ENV & MODEL --------------------
set_tracing_disabled(disabled=True)
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your environment (.env)")

# external client configured for Gemini via OpenAI-compatible path
external_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

run_config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

# -------------------- LOAD CSV (safe fill to avoid FutureWarning) --------------------
df = pd.read_csv(DATA_PATH)

for col in df.columns:
    if df[col].dtype == "object":
        # keep text columns as strings
        df[col] = df[col].fillna("").astype(str)
    else:
        # numeric columns: fillna with 0 (keeps dtype numeric)
        try:
            df[col] = df[col].fillna(0)
        except Exception:
            # fallback: turn into strings
            df[col] = df[col].astype(str).fillna("")

# ensure price numeric if present
if "price" in df.columns:
    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0
    df["price"] = df["price"].apply(safe_float)

# normalize text columns for searching
text_cols = ["proptitle", "propdesc", "temp_area", "temp_loc", "proptype", "propcat", "propfor", "furnished"]
for c in text_cols:
    if c in df.columns:
        df[c] = df[c].astype(str)

# -------------------- UTILS --------------------
def parse_human_number(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip().lower().replace(",", "")
    m = re.match(r"^([\d\.]+)\s*([kmn]*)$", s)
    if m:
        num = float(m.group(1))
        suffix = m.group(2)
        if suffix in PRICE_SCALE:
            return num * PRICE_SCALE[suffix]
        return num
    try:
        return float(s)
    except Exception:
        return None

# -------------------- TOOLS (used by the LLM) --------------------
# NOTE: Use explicit typed defaults so the function-calling schema contains type.
@function_tool
async def search_properties(
    area: str = "",
    propfor: str = "",
    bedrooms: int = 0,           # 0 means "not specified"
    furnished: str = "",         # empty string means not specified
    proptype: str = "",
    propcat: str = "",
    min_price: float = 0.0,     # 0.0 means "not specified"
    max_price: float = 0.0,     # 0.0 means "not specified"
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search and return a list (max=limit) of matching property dicts.
    Sentinels: empty string or 0/0.0 => parameter not used.
    """
    results = df.copy()

    # area (text) filter
    if area and area.strip() != "":
        a = area.lower()
        mask = pd.Series(False, index=results.index)
        for c in ["proptitle", "propdesc", "temp_area", "temp_loc"]:
            if c in results.columns:
                mask = mask | results[c].str.lower().str.contains(a, na=False)
        results = results[mask]

    # propfor: Sale / Rent
    if propfor and propfor.strip() != "":
        if "propfor" in results.columns:
            results = results[results["propfor"].str.lower() == propfor.lower()]

    # bedrooms (0 means not specified)
    if bedrooms and "bedrooms" in results.columns:
        def bed_match(x):
            try:
                return int(float(x)) == int(bedrooms)
            except Exception:
                return False
        results = results[results["bedrooms"].apply(bed_match)]

    # furnished
    if furnished and furnished.strip() != "" and "furnished" in results.columns:
        results = results[results["furnished"].str.lower().str.contains(furnished.lower(), na=False)]

    # proptype
    if proptype and proptype.strip() != "" and "proptype" in results.columns:
        results = results[results["proptype"].str.lower().str.contains(proptype.lower(), na=False)]

    # propcat
    if propcat and propcat.strip() != "" and "propcat" in results.columns:
        results = results[results["propcat"].str.lower().str.contains(propcat.lower(), na=False)]

    # price filters (0.0 means not specified)
    if min_price and "price" in results.columns:
        results = results[results["price"].astype(float) >= float(min_price)]

    if max_price and "price" in results.columns:
        results = results[results["price"].astype(float) <= float(max_price)]

    # build display columns
    display_cols = [c for c in ["propid", "proptitle", "propfor", "price", "bedrooms", "furnished", "proptype", "temp_area"] if c in results.columns]
    # convert to list of dicts
    out = results.head(limit)[display_cols].fillna("").to_dict(orient="records")

    # If no matches, explicitly return an empty list (the LLM will handle messaging)
    return out

@function_tool
async def get_price_by_id(propid: int = 0) -> Dict[str, Any]:
    """
    Return property data for given propid. propid=0 means invalid / not provided.
    """
    if not propid:
        return {"found": False}
    matches = df[df["propid"] == propid]
    if matches.empty:
        return {"found": False}
    row = matches.iloc[0]
    cols = [c for c in ["propid", "proptitle", "propfor", "price", "bedrooms", "furnished", "proptype", "temp_area"] if c in df.columns]
    return {"found": True, "data": row[cols].to_dict()}

@function_tool
async def summarize_results(results: list) -> str:
    """
    Accepts the list returned by search_properties and returns a short human summary.
    """
    if not results:
        return "No matching properties found."
    lines = []
    for r in results:
        lines.append(f"[ID {r.get('propid','-')}] {r.get('proptitle','No title')} | {r.get('propfor','')} | Price: {r.get('price','N/A')} | Beds: {r.get('bedrooms','-')} | Furnished: {r.get('furnished','-')} | Type: {r.get('proptype','-')} | Area: {r.get('temp_area','-')}")
    return "\n".join(lines)

# -------------------- AGENT (instructions and tools) --------------------
agent_instructions = (
    "You are Property Insights Assistant. You MUST only answer property-related queries. "
    "For any factual property lookup (prices, availability, listings), call the tools provided "
    "and NEVER hallucinate prices or details. Use the tools as the source of truth. "
    "If the user asks unrelated questions, politely refuse. If the user asks follow-ups (e.g., 'under 50k'), "
    "reuse the conversation context provided in CONTEXT. If clarification needed, ask concise property-specific questions."
)

agent = Agent(
    name="SciMath Assistant",
    instructions="You are a helpful assistant capable of basic and scientific calculations. You can perform arithmetic operations, trigonometric calculations, logarithms, and more. Use the tools provided to answer the user's questions.",
    tools=[
        add, subtract, multiply, average, power,
        square_root, factorial, sine, cosine, tangent, logarithm
    ]
)

# -------------------- TOP-LEVEL LOOP (Streamlit app) --------------------
st.set_page_config(page_title="Ali — Property Insights Assistant", layout="wide")
st.title("🏠 Ali — Property Insights Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("### Ask your property-related question below:")

user_q = st.text_input("Your question:", key="user_input")

if st.button("Send", key="send_btn") and user_q.strip():
    # Guardrail: quick refuse if not property
    if not is_property_query(user_q):
        assistant_reply = "Sorry — I can only help with property-related queries (search, price, rent/sale, bedrooms, furnishing, area, or property ID lookups)."
    else:
        # Build context for LLM
        context_text = ""
        if st.session_state.chat_history:
            lines = []
            for entry in st.session_state.chat_history:
                role = entry["role"]
                lines.append(f"{role.upper()}: {entry['content']}")
            context_text = "CONTEXT:\n" + "\n".join(lines) + "\n\n"
        prompt = context_text + "USER: " + user_q

        # Call the Agent (LLM) with event loop fix
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            run_result = loop.run_until_complete(Runner.run(agent, prompt, run_config=run_config))
            assistant_reply = run_result.final_output if hasattr(run_result, "final_output") else str(run_result)
            loop.close()
        except Exception as e:
            assistant_reply = f"Error while contacting model: {e}"

    # Update chat history
    st.session_state.chat_history.append({"role": "user", "content": user_q})
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
    st.rerun()

# Display chat history
st.markdown("### Conversation")
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(f"**You:** {entry['content']}")
    else:
        st.markdown(f"**Ali:** {entry['content']}")

st.markdown("---")
st.caption("Ali — uses your CSV as source of truth. Make sure GEMINI_API_KEY is set for agent access.")
