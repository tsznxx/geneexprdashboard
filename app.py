import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Gene Expression Dashboard", layout="wide")

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
import pandas as pd

def read_expression_or_meta(file_input):
    """
    Reads tabular data (.csv, .tsv, .txt) including gzipped files (.gz).
    Uses engine='python' with sep=None to automatically infer the delimiter.
    """
    # sep=None allows pandas to inspect the uncompressed stream and auto-detect '\t' vs ','
    df = pd.read_csv(
        file_input, 
        sep=None, 
        engine="python", 
        index_col=0, 
        compression="infer"
    )
    return df

def reset_app_state():
    """Clears loaded datasets and increments uploader keys to reset widgets."""
    for key in ["expr_df", "meta_df"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1

# Initialize dynamic uploader key index
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# Track overall data status
data_loaded = ("expr_df" in st.session_state) and ("meta_df" in st.session_state)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.logo("https://raw.githubusercontent.com/streamlit/streamlit/main/docs/static/img/streamlit-mark-color.png")

with st.sidebar:
    st.header("🧬 Data Management")

    if not data_loaded:
        # --- STATE A: DATA ENTRY MODE ---
        st.subheader("1. Demo Data")
        if st.button("⚡ Load Local Example Datasets", use_container_width=True):
            expr_path = "exampledata/example_expression.tsv"
            meta_path = "exampledata/example_metadata.tsv"
            
            if os.path.exists(expr_path) and os.path.exists(meta_path):
                st.session_state["expr_df"] = read_expression_or_meta(expr_path)
                st.session_state["meta_df"] = read_expression_or_meta(meta_path)
                st.rerun()
            else:
                st.error("Example datasets missing from `exampledata/` directory.")

        st.divider()

        st.subheader("2. Upload Your Data")
        # Dynamic keys ensure uploaders reset completely when resetting state
        u_key = st.session_state["uploader_key"]
        
        uploaded_expr = st.file_uploader(
            "Upload Expression Matrix (.csv, .tsv, .gz)", 
            type=["csv", "tsv", "txt", "gz"], 
            key=f"up_expr_{u_key}"
        )
        uploaded_meta = st.file_uploader(
            "Upload Metadata (.csv, .tsv, .gz)", 
            type=["csv", "tsv", "txt", "gz"], 
            key=f"up_meta_{u_key}"
        )

        if uploaded_expr and uploaded_meta:
            st.session_state["expr_df"] = read_expression_or_meta(uploaded_expr)
            st.session_state["meta_df"] = read_expression_or_meta(uploaded_meta)
            st.rerun()

    else:
        # --- STATE B: ACTIVE DATA MODE ---
        st.success("✅ Datasets Active")
        st.caption(f"**Expression**: {st.session_state['expr_df'].shape[0]} Genes × {st.session_state['expr_df'].shape[1]} Samples")
        st.caption(f"**Metadata**: {st.session_state['meta_df'].shape[0]} Samples × {st.session_state['meta_df'].shape[1]} Attributes")

        st.divider()

        # Action 1: Replace Metadata Only
        with st.popover("🔄 Replace Metadata Only", use_container_width=True):
            st.markdown("Upload new metadata while keeping the current gene expression matrix.")
            new_meta = st.file_uploader(
                "Select New Metadata File", 
                type=["csv", "tsv", "txt", "gz"],
                key="replace_meta_uploader"
            )
            if new_meta:
                st.session_state["meta_df"] = read_expression_or_meta(new_meta)
                st.success("Metadata updated successfully!")
                st.rerun()

        # Action 2: Reset / Start Over
        if st.button("🛑 Start Over / Reset", type="primary", use_container_width=True):
            reset_app_state()
            st.rerun()

        st.divider()

        # Section 3: Downloads
        st.subheader("Downloads")
        dl_expr = st.session_state["expr_df"].to_csv(sep="\t").encode("utf-8")
        dl_meta = st.session_state["meta_df"].to_csv(sep="\t").encode("utf-8")

        st.download_button(
            label="📥 Download Active Expression TSV",
            data=dl_expr,
            file_name="active_expression.tsv",
            mime="text/tab-separated-values",
            use_container_width=True
        )
        st.download_button(
            label="📥 Download Active Metadata TSV",
            data=dl_meta,
            file_name="active_metadata.tsv",
            mime="text/tab-separated-values",
            use_container_width=True
        )

# ---------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# ---------------------------------------------------------
st.title("Gene Expression Dashboard")

if not data_loaded:
    st.info("👈 Please load example data or upload your expression & metadata files via the sidebar.")
    st.stop()

# Main app dashboard tabs continue here...
tab_overview, tab_pca = st.tabs(["📋 Data Preview", "📊 PCA Analysis"])

with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Expression Matrix")
        st.dataframe(st.session_state["expr_df"].head(10), use_container_width=True)
    with c2:
        st.subheader("Metadata")
        st.dataframe(st.session_state["meta_df"].head(10), use_container_width=True)
