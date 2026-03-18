"""Hackathon-ready Streamlit frontend for the AI Research Agent."""

from __future__ import annotations

import copy

import streamlit as st

from main import load_config
from src.frontend.dashboard_service import (
    answer_query,
    get_dashboard_status,
    run_ingest,
    run_train,
    save_uploaded_pdfs,
)


st.set_page_config(
    page_title="AI Research Agent",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    """Apply a strong custom visual identity to the Streamlit app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #0b1020;
            --panel: rgba(13, 22, 45, 0.72);
            --panel-strong: rgba(17, 27, 56, 0.92);
            --text: #f5f7ff;
            --muted: #9aa6c8;
            --line: rgba(255, 255, 255, 0.08);
            --aqua: #75f7d4;
            --blue: #4d7cff;
            --amber: #ffb84d;
            --rose: #ff7d8b;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(77,124,255,0.28), transparent 28%),
                radial-gradient(circle at 90% 0%, rgba(117,247,212,0.16), transparent 25%),
                linear-gradient(135deg, #06101f 0%, #0b1020 40%, #120f1d 100%);
            color: var(--text);
            font-family: "Space Grotesk", sans-serif;
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.03em;
        }

        [data-testid="stSidebar"] {
            background: rgba(7, 12, 26, 0.88);
            border-right: 1px solid var(--line);
        }

        .hero {
            background: linear-gradient(135deg, rgba(77,124,255,0.18), rgba(117,247,212,0.08));
            border: 1px solid rgba(117,247,212,0.16);
            border-radius: 28px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
        }

        .eyebrow {
            display: inline-block;
            color: var(--aqua);
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .subcopy {
            color: var(--muted);
            max-width: 56rem;
            font-size: 1rem;
            line-height: 1.55;
        }

        .card {
            background: var(--panel);
            backdrop-filter: blur(10px);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1rem 0.9rem 1rem;
            min-height: 150px;
        }

        .metric-value {
            font-size: 2.1rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.88rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-glow-aqua { color: var(--aqua); }
        .metric-glow-blue { color: #8fb1ff; }
        .metric-glow-amber { color: var(--amber); }
        .metric-glow-rose { color: #ff9fb3; }

        .section-shell {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1rem 1rem 0.4rem 1rem;
            margin-top: 1rem;
        }

        .answer-box {
            background: linear-gradient(180deg, rgba(117,247,212,0.07), rgba(255,255,255,0.02));
            border: 1px solid rgba(117,247,212,0.18);
            border-radius: 22px;
            padding: 1rem 1rem 0.9rem 1rem;
            margin-top: 0.8rem;
        }

        .overlay-wrap {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.62);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 999999;
            backdrop-filter: blur(7px);
        }

        .overlay-card {
            width: min(520px, 86vw);
            background: rgba(7, 14, 31, 0.92);
            border: 1px solid rgba(117,247,212,0.18);
            box-shadow: 0 30px 90px rgba(0, 0, 0, 0.4);
            border-radius: 28px;
            padding: 1.6rem 1.6rem 1.35rem 1.6rem;
            text-align: center;
        }

        .overlay-spinner {
            width: 70px;
            height: 70px;
            margin: 0 auto 1rem auto;
            border-radius: 50%;
            border: 4px solid rgba(255,255,255,0.08);
            border-top-color: var(--aqua);
            border-right-color: var(--blue);
            animation: spinPulse 0.95s linear infinite;
        }

        .overlay-title {
            color: var(--text);
            font-weight: 700;
            font-size: 1.1rem;
            margin-bottom: 0.45rem;
        }

        .overlay-sub {
            color: var(--muted);
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.86rem;
            letter-spacing: 0.03em;
        }

        @keyframes spinPulse {
            to { transform: rotate(360deg); }
        }

        .tiny {
            color: var(--muted);
            font-size: 0.85rem;
            font-family: "IBM Plex Mono", monospace;
        }

        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.02);
        }

        .stButton > button {
            border-radius: 999px;
            font-weight: 700;
            border: 1px solid rgba(117,247,212,0.2);
            background: linear-gradient(135deg, rgba(77,124,255,0.22), rgba(117,247,212,0.16));
            color: white;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.03);
            border-radius: 999px;
            padding: 0.55rem 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str, tone: str):
    """Render a styled metric card."""
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value metric-glow-{tone}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_runtime_config(base_config: dict, provider: str, model_name: str):
    """Build a runtime config from sidebar controls."""
    config = copy.deepcopy(base_config)
    config["mode"] = "hybrid"
    config["llm_provider"] = provider
    if provider == "ollama":
        config["ollama_model"] = model_name
    elif provider == "openai":
        config["openai_model"] = model_name
    elif provider == "google":
        config["google_model"] = model_name
    return config


def render_overlay(target, title: str, subtitle: str):
    """Render a full-screen translucent working overlay."""
    target.markdown(
        f"""
        <div class="overlay-wrap">
            <div class="overlay-card">
                <div class="overlay-spinner"></div>
                <div class="overlay-title">{title}</div>
                <div class="overlay-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_styles()
    base_config = load_config()

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Hackathon Control Surface</div>
            <h1 style="margin:0 0 0.35rem 0;">AI Research Agent</h1>
            <div class="subcopy">
                Upload PDFs. Ingest them. Train your retriever. Interrogate the corpus.
                Flip between grounded document QA and free-form LLM chat when the docs come up short.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Model Control")
        provider = st.selectbox("LLM Provider", ["ollama", "openai", "google"], index=0)
        default_model = {
            "ollama": base_config.get("ollama_model", "phi3:mini"),
            "openai": base_config.get("openai_model", "gpt-4o-mini"),
            "google": base_config.get("google_model", "gemini-1.5-flash"),
        }[provider]
        model_name = st.text_input("Model Name", value=default_model)
        use_general_fallback = st.toggle("Fallback to general LLM chat when docs fail", value=True)
        max_context = st.slider(
            "LLM Context Budget",
            1000,
            8000,
            int(base_config.get("max_context_tokens", 4000)),
            250,
        )

        st.header("Retrieval Tuning")
        top_k = st.slider("Top K", 1, 10, int(base_config.get("top_k", 5)))
        threshold = st.slider(
            "Similarity Threshold",
            0.0,
            0.9,
            float(base_config.get("similarity_threshold", 0.3)),
            0.05,
        )

    runtime_config = make_runtime_config(base_config, provider, model_name)
    runtime_config["top_k"] = top_k
    runtime_config["similarity_threshold"] = threshold
    runtime_config["max_context_tokens"] = max_context

    status = get_dashboard_status(runtime_config)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric("PDF Library", str(status["pdf_count"]), "aqua")
    with col2:
        render_metric("Indexed Chunks", str(status["chunk_count"]), "blue")
    with col3:
        render_metric("Vector Store", "Ready" if status["vector_store_ready"] else "Cold", "amber")
    with col4:
        render_metric("Encoder", "Trained" if status["trained_model_ready"] else "Base", "rose")

    upload_tab, train_tab, ask_tab = st.tabs(["Drop Zone", "Train Lab", "Ask Anything"])

    with upload_tab:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        left, right = st.columns([1.1, 1.3])
        with left:
            st.subheader("Document Intake")
            ingest_overlay = st.empty()
            uploaded_files = st.file_uploader(
                "Upload one or more PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                help="Files are saved directly into the project's data/pdfs folder.",
            )
            if st.button("Save PDFs to Library", use_container_width=True, disabled=not uploaded_files):
                saved_paths = save_uploaded_pdfs(uploaded_files, runtime_config["pdf_directory"])
                st.success(f"Saved {len(saved_paths)} file(s) into {runtime_config['pdf_directory']}.")
                st.rerun()

            if st.button("Ingest Library", use_container_width=True):
                progress_box = st.empty()

                def on_ingest_progress(message: str):
                    render_overlay(ingest_overlay, "Ingesting document library", message)
                    progress_box.caption(message)

                result = run_ingest(runtime_config, progress_callback=on_ingest_progress)
                ingest_overlay.empty()
                if result.ok:
                    st.success(result.message)
                    if result.stats:
                        st.caption(
                            f"Processed {result.stats['documents']} pages into {result.stats['chunks']} chunks across {result.stats['sources']} source files."
                        )
                else:
                    st.error(result.message)
                st.code(result.logs or "No logs captured.", language="text")

        with right:
            st.subheader("Current PDF Library")
            if status["pdf_files"]:
                for pdf_name in status["pdf_files"]:
                    st.markdown(f"- `{pdf_name}`")
            else:
                st.info("No PDFs found yet. Upload some fuel for the retriever.")

            st.markdown(
                """
                <div class="tiny">
                Flow: upload -> ingest -> train -> ask.  
                Ingest rebuilds the chunk store and vector index from the PDFs currently inside <code>data/pdfs</code>.
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with train_tab:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        left, right = st.columns([1.1, 1.1])
        with left:
            st.subheader("Retriever Training")
            train_overlay = st.empty()
            st.write(
                "Train the custom embedding encoder on your processed chunks. "
                "This improves semantic retrieval quality over plain fallback search."
            )
            if st.button("Train Encoder", use_container_width=True):
                progress_box = st.empty()

                def on_train_progress(message: str):
                    render_overlay(train_overlay, "Training retrieval model", message)
                    progress_box.caption(message)

                result = run_train(runtime_config, progress_callback=on_train_progress)
                train_overlay.empty()
                if result.ok:
                    st.success(result.message)
                else:
                    st.error(result.message)
                st.code(result.logs or "No logs captured.", language="text")

        with right:
            st.subheader("Live Config Snapshot")
            st.json(
                {
                    "provider": runtime_config["llm_provider"],
                    "model": {
                        "ollama": runtime_config.get("ollama_model"),
                        "openai": runtime_config.get("openai_model"),
                        "google": runtime_config.get("google_model"),
                    }[runtime_config["llm_provider"]],
                    "top_k": runtime_config["top_k"],
                    "similarity_threshold": runtime_config["similarity_threshold"],
                    "chunk_size": runtime_config["chunk_size"],
                    "chunk_overlap": runtime_config["chunk_overlap"],
                    "epochs": runtime_config["epochs"],
                }
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with ask_tab:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        st.subheader("RAG Cockpit")
        query = st.text_area(
            "Ask a question or drop a normal prompt",
            placeholder="What authorization methods does API Gateway support? Or: explain RAG like I'm five.",
            height=120,
        )
        source_options = ["All documents"] + status["pdf_files"]
        selected_source = st.selectbox("Source Filter", source_options, index=0)
        interaction_mode = st.radio(
            "Response Mode",
            ["Document-grounded RAG", "Pure LLM chat"],
            horizontal=True,
        )

        if st.button("Generate Answer", type="primary", use_container_width=True, disabled=not query.strip()):
            with st.spinner("Interrogating the stack..."):
                response = answer_query(
                    runtime_config,
                    query.strip(),
                    source_filter=None if selected_source == "All documents" else selected_source,
                    use_general_llm=interaction_mode == "Pure LLM chat",
                    fallback_to_general=use_general_fallback,
                )

            mode_labels = {
                "rag_llm": "RAG + LLM synthesis",
                "extractive": "Extractive grounded answer",
                "general_llm": "Pure LLM chat",
                "general_llm_fallback": "LLM fallback because the docs were cold",
                "not_found": "No answer found in the indexed documents",
                "failed_general_llm": "General chat failed",
            }
            st.markdown(
                f"""
                <div class="answer-box">
                    <div class="tiny">Mode: {mode_labels.get(response['mode'], response['mode'])}</div>
                    <div style="font-size:1.05rem; line-height:1.65; margin-top:0.55rem;">{response['answer']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if response["sources"]:
                st.markdown("**Sources**")
                for source in response["sources"]:
                    st.markdown(f"- {source}")

            if response["results"]:
                with st.expander("Inspect retrieved chunks"):
                    for idx, item in enumerate(response["results"], 1):
                        st.markdown(
                            f"**Result {idx}** - `{item['source']}` - page {item['page']} - score `{item['score']:.3f}`"
                        )
                        st.write(item["text"][:1000])
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
