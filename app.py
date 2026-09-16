import json
import os
from typing import Any, Dict, Optional

import streamlit as st

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

APP_TITLE = "📱 MobileSpec AI"
MODEL_NAME = "gemini-2.5-flash"
MAX_IMAGE_BYTES = 20 * 1024 * 1024

SPEC_FIELDS = [
    ("brand", "Brand"),
    ("model", "Model"),
    ("release_date", "Release date / year"),
    ("display_size", "Display size"),
    ("display_type", "Display type"),
    ("display_resolution", "Display resolution"),
    ("refresh_rate", "Refresh rate"),
    ("processor", "Processor / chipset"),
    ("ram", "RAM"),
    ("storage", "Internal storage"),
    ("rear_camera", "Rear camera"),
    ("front_camera", "Front camera"),
    ("battery", "Battery capacity"),
    ("charging", "Charging"),
    ("os", "Operating system"),
    ("network", "5G / network support"),
    ("sim", "SIM type"),
    ("dimensions", "Dimensions"),
    ("weight", "Weight"),
    ("water_resistance", "Water / dust resistance"),
    ("colors", "Colors"),
    ("other", "Other important specifications"),
]

SPEC_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "confidence": {
            "type": "STRING",
            "description": "Overall identification/specification confidence: High, Medium, or Low.",
        },
        "identified_model": {
            "type": "STRING",
            "description": "Best identified phone model. Use 'Unknown' if it cannot be identified.",
        },
        "variant_notes": {
            "type": "STRING",
            "description": "Regional, RAM/storage, carrier, or other variant differences. Use 'None' when not known.",
        },
        "specifications": {
            "type": "OBJECT",
            "properties": {
                key: {
                    "type": "OBJECT",
                    "properties": {
                        "value": {
                            "type": "STRING",
                            "description": "Specification value. Use 'Not available / Uncertain' if not reliably known.",
                        },
                        "status": {
                            "type": "STRING",
                            "description": "One of: Confirmed, Variant-dependent, or Uncertain.",
                        },
                    },
                    "required": ["value", "status"],
                }
                for key, _ in SPEC_FIELDS
            },
            "required": [key for key, _ in SPEC_FIELDS],
        },
        "analysis": {
            "type": "OBJECT",
            "properties": {
                "key_features": {"type": "ARRAY", "items": {"type": "STRING"}},
                "performance": {"type": "STRING"},
                "camera": {"type": "STRING"},
                "battery": {"type": "STRING"},
                "display": {"type": "STRING"},
                "pros": {"type": "ARRAY", "items": {"type": "STRING"}},
                "limitations": {"type": "ARRAY", "items": {"type": "STRING"}},
                "use_cases": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": [
                "key_features",
                "performance",
                "camera",
                "battery",
                "display",
                "pros",
                "limitations",
                "use_cases",
            ],
        },
    },
    "required": ["confidence", "identified_model", "variant_notes", "specifications", "analysis"],
}

SYSTEM_INSTRUCTION = """
You are a careful mobile-phone specification research assistant.

Your job is to identify the exact smartphone model and provide reliable specifications.
You may use Google Search grounding when available. Prefer manufacturer pages and other
reputable sources. Do not invent or guess specifications.

Rules:
1. If a specification cannot be verified with reasonable confidence, return exactly
   'Not available / Uncertain' for its value and set status to 'Uncertain'.
2. If a value differs by region, carrier, RAM/storage configuration, or market, explain
   that in the value or variant_notes and use status 'Variant-dependent'.
3. For an image, identify the phone only when the visual evidence supports it. Do not
   infer an exact model merely from a generic-looking phone design.
4. If the exact model cannot be established, set identified_model to 'Unknown' and keep
   uncertain fields explicit.
5. Do not silently substitute a different model with a similar name.
6. Keep analysis factual and concise. Do not make unsupported claims about performance.
7. Return JSON matching the supplied schema and nothing else.
"""


def get_api_key() -> Optional[str]:
    """Read the Gemini key from Streamlit secrets first, then the environment."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass
    key = os.getenv("GEMINI_API_KEY")
    return key.strip() if key else None


@st.cache_resource(show_spinner=False)
def make_client(api_key: str):
    if genai is None:
        raise RuntimeError("google-genai is not installed. Run: pip install -r requirements.txt")
    return genai.Client(api_key=api_key)


def build_config():
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=SPEC_SCHEMA,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )


def parse_response(response: Any) -> Dict[str, Any]:
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned an empty response.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini returned a response that was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError("Gemini returned an unexpected JSON structure.")
    return data


def call_gemini(contents: Any) -> Dict[str, Any]:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to Streamlit Secrets or your environment."
        )
    client = make_client(api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=build_config(),
    )
    return parse_response(response)


def normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing fields defensively so UI rendering never crashes."""
    data.setdefault("confidence", "Uncertain")
    data.setdefault("identified_model", "Unknown")
    data.setdefault("variant_notes", "None")
    specs = data.setdefault("specifications", {})
    for key, _ in SPEC_FIELDS:
        item = specs.setdefault(key, {})
        if not isinstance(item, dict):
            item = {"value": str(item), "status": "Uncertain"}
            specs[key] = item
        item.setdefault("value", "Not available / Uncertain")
        item.setdefault("status", "Uncertain")
    analysis = data.setdefault("analysis", {})
    for key in ["performance", "camera", "battery", "display"]:
        analysis.setdefault(key, "Not available / Uncertain")
    for key in ["key_features", "pros", "limitations", "use_cases"]:
        value = analysis.setdefault(key, [])
        if not isinstance(value, list):
            analysis[key] = [str(value)]
    return data


def display_result(data: Dict[str, Any]):
    data = normalize_result(data)
    st.session_state["result"] = data
    st.session_state["last_model"] = data.get("identified_model", "Unknown")


def render_specs(data: Dict[str, Any]):
    data = normalize_result(data)
    st.subheader("📋 Specifications")
    specs = data["specifications"]
    rows = []
    for key, label in SPEC_FIELDS:
        item = specs.get(key, {})
        rows.append(
            {
                "Specification": label,
                "Value": item.get("value", "Not available / Uncertain"),
                "Status": item.get("status", "Uncertain"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.caption(f"Overall confidence: {data.get('confidence', 'Uncertain')}")
    notes = data.get("variant_notes", "None")
    if notes and notes != "None":
        st.info(f"Variant / regional notes: {notes}")


def render_analysis(data: Dict[str, Any]):
    data = normalize_result(data)
    analysis = data["analysis"]
    st.subheader("🤖 AI Analysis")
    for title, key in [
        ("Key features", "key_features"),
        ("Pros", "pros"),
        ("Limitations", "limitations"),
        ("Suitable use cases", "use_cases"),
    ]:
        st.markdown(f"**{title}**")
        values = analysis.get(key, [])
        if values:
            for value in values:
                st.write(f"• {value}")
        else:
            st.write("Not available / Uncertain")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Performance**")
        st.write(analysis.get("performance", "Not available / Uncertain"))
        st.markdown("**Camera**")
        st.write(analysis.get("camera", "Not available / Uncertain"))
    with col2:
        st.markdown("**Battery**")
        st.write(analysis.get("battery", "Not available / Uncertain"))
        st.markdown("**Display**")
        st.write(analysis.get("display", "Not available / Uncertain"))


def show_error(exc: Exception):
    message = str(exc)
    lower = message.lower()
    if "api key" in lower or "gemini_api_key" in lower:
        st.error(message)
        st.info("Configure GEMINI_API_KEY in .streamlit/secrets.toml locally or in Streamlit Cloud Secrets.")
    elif "429" in lower or "rate" in lower or "quota" in lower:
        st.error("Gemini API rate limit or quota was reached.")
        st.info(
            "This app uses Google Search grounding for phone-spec verification. "
            "Gemini 2.5 Flash supports Search grounding on the Free Tier with a shared "
            "limit of up to 500 grounded requests per day. If the limit has been reached, "
            "wait for the quota to reset or check usage in Google AI Studio."
        )
    elif "401" in lower or "403" in lower or "permission" in lower or "unauth" in lower:
        st.error("Gemini API authentication/permission failed. Check that your API key is valid and enabled.")
    elif "json" in lower:
        st.error("Gemini returned an unexpected response format. Please try again.")
    else:
        st.error(f"Gemini request failed: {message}")


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="MobileSpec AI", page_icon="📱", layout="wide")

st.title(APP_TITLE)
st.caption("Identify a phone from its model name or an uploaded image, then inspect grounded specifications.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"Gemini model: `{MODEL_NAME}`")
    if get_api_key():
        st.success("Gemini API key detected")
    else:
        st.warning("Gemini API key not configured")
    st.divider()
    st.write("**Security:** API keys are read from Streamlit Secrets/environment variables and are never stored in this app's source code.")

# Keep the most recent result available across Streamlit reruns.
result = st.session_state.get("result")

input_tab, image_tab, specs_tab, analysis_tab, status_tab = st.tabs(
    ["🔎 Enter Model", "📷 Upload Image", "📋 Specifications", "🤖 AI Analysis", "🔌 API Status"]
)

with input_tab:
    st.header("Enter Mobile Model")
    model_input = st.text_input(
        "Mobile model",
        placeholder="e.g. Samsung Galaxy S25 Ultra",
        key="model_input",
    )
    if st.button("Analyze / Get Specifications", type="primary", use_container_width=True):
        if not model_input.strip():
            st.warning("Please enter a mobile model first.")
        else:
            prompt = f"Analyze the smartphone model: {model_input.strip()}. Verify the exact model and provide its specifications and concise analysis."
            with st.spinner("Searching and analyzing the phone specifications..."):
                try:
                    display_result(call_gemini(prompt))
                    st.success(f"Analysis completed for: {st.session_state.get('last_model', model_input.strip())}")
                except Exception as exc:
                    show_error(exc)

    if result:
        st.info(f"Current result: **{result.get('identified_model', 'Unknown')}**")

with image_tab:
    st.header("Upload Mobile Image")
    uploaded = st.file_uploader(
        "Upload a clear photo of the mobile phone",
        type=["jpg", "jpeg", "png", "webp"],
        help="For best identification, use a clear image showing the front, back, camera module, or visible branding/model markings.",
    )
    if uploaded:
        if uploaded.size > MAX_IMAGE_BYTES:
            st.error("Image is larger than 20 MB. Please upload a smaller image.")
        else:
            st.image(uploaded, caption="Uploaded phone image", use_container_width=True)
            if st.button("Identify & Analyze Image", type="primary", use_container_width=True):
                image_bytes = uploaded.getvalue()
                mime_type = uploaded.type or "image/jpeg"
                image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                prompt = """
Identify the smartphone shown in this image if the visual evidence supports an exact model.
Look carefully at visible branding, camera layout, buttons, dimensions, distinctive design,
and any readable text. If several models are visually similar and the image is insufficient,
return Unknown or Low confidence rather than guessing. Then provide the specifications and
concise analysis for the identified model.
"""
                with st.spinner("Identifying the phone image and checking specifications..."):
                    try:
                        display_result(call_gemini([image_part, prompt]))
                        confidence = st.session_state["result"].get("confidence", "Uncertain")
                        model = st.session_state["result"].get("identified_model", "Unknown")
                        if model == "Unknown" or confidence.lower() == "low":
                            st.warning("The exact model could not be identified confidently from this image.")
                        else:
                            st.success(f"Identified: {model} ({confidence} confidence)")
                    except Exception as exc:
                        show_error(exc)

with specs_tab:
    st.header("Specifications")
    if result:
        render_specs(result)
    else:
        st.info("Analyze a model or upload an image first.")

with analysis_tab:
    st.header("AI Analysis")
    if result:
        render_analysis(result)
    else:
        st.info("Analyze a model or upload an image first.")

with status_tab:
    st.header("API Status")
    st.write(f"Configured Gemini model: `{MODEL_NAME}`")
    st.caption("Phone analysis uses Google Search grounding to verify current specifications.")
    key = get_api_key()
    if key:
        st.success("GEMINI_API_KEY is configured.")
        if st.button("Test Gemini API Connection", use_container_width=True):
            with st.spinner("Testing Gemini API..."):
                try:
                    client = make_client(key)
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents="Reply with exactly: API connection successful.",
                        config=types.GenerateContentConfig(max_output_tokens=20),
                    )
                    st.success("Gemini API connection successful.")
                    st.code(response.text or "No text returned")
                except Exception as exc:
                    show_error(exc)
    else:
        st.error("GEMINI_API_KEY is not configured.")
        st.markdown("""
**Local setup**

Create `.streamlit/secrets.toml` in the project folder:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Do **not** commit this file to GitHub.
""")

st.divider()
st.caption("Information is AI-generated and should be checked against official manufacturer specifications, especially for regional variants.")
