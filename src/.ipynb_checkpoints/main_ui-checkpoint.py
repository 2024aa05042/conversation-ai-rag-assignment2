"""
Part 1.5 COMPLETE Hybrid RAG UI - Streamlit (FULLY DOCUMENTED).
Displays: query → answer + chunks (dense/sparse/RRF scores/time).

Author: NLP Expert Assistant
Purpose: Assignment Part 1.5 - Interactive demo of full Hybrid RAG pipeline
Requirements: Part 1.1-1.4 modules + chunks.json + streamlit
"""

# =====================================================
# CORE IMPORTS
# =====================================================
import streamlit as st                    # Web UI framework (interactive widgets, layouts)
import time                               # Timing measurements for performance metrics
from response_generation import ResponseGenerator  # Part 1.4: Full RAG pipeline (retrieve→fuse→generate)

# =====================================================
# PAGE CONFIGURATION (BROWSER TAB + LAYOUT)
# =====================================================
st.set_page_config(                       # Sets browser tab title + wide layout for tables
    page_title="Hybrid RAG Demo",         # Appears in browser tab
    layout="wide"                         # Maximizes horizontal space for answer + metrics
)

# =====================================================
# MAIN HEADER (BRANDING)
# =====================================================
st.title("🧠 Hybrid RAG System (Part 1 Complete)")  # Large title with emoji
st.markdown("**Dense (Embeddings) + Sparse (BM25) + RRF + Flan-T5**")  # Subtitle explaining pipeline

# =====================================================
# INITIALIZATION (LAZY LOADING)
# =====================================================
@st.cache_resource                        # CACHE: Loads ONCE, reuses across sessions (saves 2min!)
def load_generator():
    """Initialize ResponseGenerator (1.1→1.4 pipeline) - cached for speed."""
    return ResponseGenerator()

generator = load_generator()              # Single global instance (dense+sparse+LLM ready)

# =====================================================
# SIDEBAR: USER CONFIGURATION
# =====================================================
st.sidebar.header("⚙️ Config")           # Sidebar header (left panel)
top_n = st.sidebar.slider(                # Interactive slider widget
    "Top N Chunks",                       # Label
    3, 8, 5,                              # min, max, default, step=1
    help="Number of RRF-fused chunks fed to LLM"  # Tooltip
)

# =====================================================
# MAIN INPUT: QUERY BOX
# =====================================================
query = st.text_input(                    # Single-line text input
    "Enter your question:",               # Placeholder label
    "What is quantum mechanics?",         # Default example query
    help="Ask anything about your Wikipedia corpus!"  # Hint
)

# =====================================================
# MAIN LOGIC: GENERATE BUTTON + PROCESSING
# =====================================================
if st.button("🔍 Generate Answer", type="primary"):  # Primary button (blue, prominent)
    if query.strip():                     # Validate non-empty query
        # SPINNER: Visual feedback during processing (1-3s)
        with st.spinner("Retrieving + Generating..."):
            start = time.time()              # Precise timing start
            result = generator.generate(query, top_n=top_n)  # FULL PIPELINE: 1.1→1.4
            elapsed = time.time() - start    # End-to-end latency
        
        # =====================================================
        # LAYOUT 1: ANSWER + METRICS (2-Column)
        # =====================================================
        col1, col2 = st.columns([3, 1])   # Split: 75% answer | 25% metrics
        
        with col1:                        # LEFT: Main answer display
            st.markdown("### 🤖 **Answer**")  # Section header
            st.write(result['answer'])     # LLM generated response (markdown rendered)
        
        with col2:                        # RIGHT: Performance metric
            # METRIC: Large KPI-style display (green if improved)
            st.metric(
                "Response Time", 
                f"{result['response_time']}",  # From generator (pre-formatted)
                help="End-to-end: retrieve + fuse + generate"
            )
        
        # =====================================================
        # LAYOUT 2: RETRIEVED CHUNKS TABLE (RRF Ranked)
        # =====================================================
        st.markdown("### 📚 **Retrieved Contexts (RRF Ranked)**")
        
        # Transform results → DataFrame format (table-ready)
        df_data = []
        for i, chunk in enumerate(result['fused_chunks'], 1):  # 1-based ranking
            df_data.append({
                'Rank': i,                                    # Final RRF position
                'RRF Score': f"{chunk['rrf_score']:.4f}",    # Fusion score (0.01-0.05 typical)
                'Dense Rank': chunk['dense_rank'] or '-',    # Part 1.1 rank (or missing)
                'Sparse Rank': chunk['sparse_rank'] or '-',  # Part 1.2 rank (or missing)
                'Title': chunk['title'][:40] + "...",       # Truncated wiki title
                'Source': chunk['url'].split('/')[-1]       # Clean URL → page name
                             .replace('_', ' ')[:40],    # Underscores → spaces
                'Preview': chunk['text']                      # First 200 chars
            })
        
        # RENDER TABLE: Interactive, sortable, full-width
        st.dataframe(
            df_data, 
            use_container_width=True,     # Stretches to fill page
            hide_index=True               # No row numbers
        )
        
        # =====================================================
        # LAYOUT 3: DEBUG EXPANDER (ADVANCED USERS)
        # =====================================================
        with st.expander("🔍 Debug: Full Prompt"):  # Collapsible section
            st.text(result['full_prompt'][:1000])  # Truncated prompt (context+query)

# =====================================================
# PERMANENT HELP SECTION (EXPANDED BY DEFAULT)
# =====================================================
with st.expander("📋 Setup & Run Instructions", expanded=False):
    st.code("""
# Terminal commands:
cd ~/Downloads/hybrid_rag_final
source venv/bin/activate
streamlit run src/main_ui.py
    """, language="bash")
    st.info("🌐 **Hosts at: http://localhost:8501**")
    st.markdown("""
    **Dependencies**: sentence-transformers, faiss-cpu, rank-bm25, transformers, streamlit
    **Data**: data/chunks.json (200+ Wikipedia chunks)
    **Pipeline**: Query → Dense(1.1) + Sparse(1.2) → RRF(1.3) → Flan-T5(1.4) → UI(1.5)
    """)
