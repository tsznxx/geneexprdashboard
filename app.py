import io
import os
import gzip
import numpy as np
import pandas as pd
import plotly.express as px
import dash_bio as dashbio
import streamlit as st
from sklearn.decomposition import PCA
from scipy import stats

# Configure page layout
st.set_page_config(
    page_title="Gene Expression Dashboard",
    page_icon="🧬",
    layout="wide"
)

# ---------------------------------------------------------
# HELPER FUNCTIONS FOR FILE LOADING & DEG
# ---------------------------------------------------------
def detect_delimiter(file_name_or_obj):
    """Detects delimiter based on file extension."""
    name = getattr(file_name_or_obj, "name", str(file_name_or_obj))
    if ".tsv" in name or ".txt" in name:
        return "\t"
    return ","

def read_expression_or_meta(file_input):
    """
    Reads tabular data (.csv, .tsv, .txt) including gzipped files (.gz).
    Accepts either a file path (str) or a Streamlit UploadedFile object.
    """
    sep = detect_delimiter(file_input)
    # Pandas read_csv natively auto-detects 'gzip' compression when passing file paths or buffer objects
    df = pd.read_csv(file_input, sep=sep, index_col=0, compression="infer")
    return df

def compute_deg(expr_df, meta_df, condition_col, group_a, group_b):
    """Computes Fold Change and p-values between two sample groups."""
    samples_a = meta_df[meta_df[condition_col] == group_a].index
    samples_b = meta_df[meta_df[condition_col] == group_b].index
    
    # Restrict to samples present in expression matrix
    samples_a = [s for s in samples_a if s in expr_df.columns]
    samples_b = [s for s in samples_b if s in expr_df.columns]
    
    results = []
    for gene in expr_df.index:
        vals_a = expr_df.loc[gene, samples_a]
        vals_b = expr_df.loc[gene, samples_b]
        
        log2fc = vals_a.mean() - vals_b.mean()
        ttest_res = stats.ttest_ind(vals_a, vals_b)
        
        results.append({
            "gene": gene,
            "EFFECTSIZE": log2fc,  # Log2 Fold Change
            "P": ttest_res.pvalue if not np.isnan(ttest_res.pvalue) else 1.0
        })
        
    return pd.DataFrame(results)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & DATA LOADING
# ---------------------------------------------------------
st.logo("https://raw.githubusercontent.com/streamlit/streamlit/main/docs/static/img/streamlit-mark-color.png")

with st.sidebar:
    st.header("🧬 Data Management")
    
    # Section 1: Load Example Data from 'exampledata' folder
    st.subheader("1. Demo Data")
    if st.button("⚡ Load Local Example Datasets", use_container_width=True):
        # Look for files inside exampledata folder (supports plain or .gz)
        expr_path = None
        meta_path = None
        
        # Priority order for detection inside exampledata/
        possible_expr = ["exampledata/example_expression.tsv", "exampledata/example_expression.tsv.gz",
                         "exampledata/example_expression.csv", "exampledata/example_expression.csv.gz",
                         "exampledata/example_expression.txt", "exampledata/example_expression.txt.gz"]
        possible_meta = ["exampledata/example_metadata.tsv", "exampledata/example_metadata.tsv.gz",
                         "exampledata/example_metadata.csv", "exampledata/example_metadata.csv.gz",
                         "exampledata/example_metadata.txt", "exampledata/example_metadata.txt.gz"]
        
        for path in possible_expr:
            if os.path.exists(path):
                expr_path = path
                break
                
        for path in possible_meta:
            if os.path.exists(path):
                meta_path = path
                break
                
        if expr_path and meta_path:
            st.session_state["expr_df"] = read_expression_or_meta(expr_path)
            st.session_state["meta_df"] = read_expression_or_meta(meta_path)
            st.success("Successfully loaded local example dataset!")
        else:
            st.error("Could not find example datasets in `exampledata/` folder.")

    st.divider()

    # Section 2: Custom Data Upload (.csv, .tsv, .txt, .gz)
    st.subheader("2. Upload Your Data")
    
    uploaded_expr = st.file_uploader(
        "Upload Expression Matrix", 
        type=["csv", "tsv", "txt", "gz"], 
        key="up_expr"
    )
    if uploaded_expr:
        st.session_state["expr_df"] = read_expression_or_meta(uploaded_expr)
        st.caption(f"Loaded: `{uploaded_expr.name}`")

    uploaded_meta = st.file_uploader(
        "Upload Metadata", 
        type=["csv", "tsv", "txt", "gz"], 
        key="up_meta"
    )
    if uploaded_meta:
        st.session_state["meta_df"] = read_expression_or_meta(uploaded_meta)
        st.caption(f"Loaded: `{uploaded_meta.name}`")

    st.divider()

    # Section 3: Downloads
    st.subheader("3. Downloads")
    
    # Download helper buffers
    if "expr_df" in st.session_state and "meta_df" in st.session_state:
        dl_expr = st.session_state["expr_df"].to_csv(sep="\t").encode("utf-8")
        dl_meta = st.session_state["meta_df"].to_csv(sep="\t").encode("utf-8")
    else:
        dl_expr = b"Gene\tSample_1\tSample_2\nBRCA1\t10.1\t8.2\n"
        dl_meta = b"SampleID\tCondition\nSample_1\tControl\nSample_2\tTreated\n"

    st.download_button(
        label="📥 Download Expression TSV",
        data=dl_expr,
        file_name="example_expression.tsv",
        mime="text/tab-separated-values",
        use_container_width=True
    )
    
    st.download_button(
        label="📥 Download Metadata TSV",
        data=dl_meta,
        file_name="example_metadata.tsv",
        mime="text/tab-separated-values",
        use_container_width=True
    )

# ---------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# ---------------------------------------------------------
st.title("Gene Expression Dashboard")

if "expr_df" not in st.session_state or "meta_df" not in st.session_state:
    st.info("👈 Please load the local example data or upload your files via the sidebar to begin.")
    st.stop()

expr_df = st.session_state["expr_df"]
meta_df = st.session_state["meta_df"]

# Create Dashboard Tabs
tab_overview, tab_pca, tab_volcano, tab_gene = st.tabs([
    "📋 Data Preview", 
    "📊 PCA Analysis", 
    "🌋 Differential Expression (Volcano)", 
    "🔍 Gene Inspector"
])

# --- TAB 1: DATA OVERVIEW ---
with tab_overview:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Expression Matrix Preview (Genes × Samples)")
        st.caption(f"Shape: {expr_df.shape[0]} Genes × {expr_df.shape[1]} Samples")
        st.dataframe(expr_df.head(10), use_container_width=True)
    with col2:
        st.subheader("Metadata Preview (Samples × Features)")
        st.caption(f"Shape: {meta_df.shape[0]} Samples × {meta_df.shape[1]} Features")
        st.dataframe(meta_df.head(10), use_container_width=True)

# --- TAB 2: PCA ANALYSIS ---
with tab_pca:
    st.subheader("Principal Component Analysis (PCA)")
    
    # Align matrix samples with metadata rows
    common_samples = [s for s in expr_df.columns if s in meta_df.index]
    
    if len(common_samples) > 1:
        sub_expr = expr_df[common_samples]
        sub_meta = meta_df.loc[common_samples]
        
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(sub_expr.T)
        
        pca_df = pd.DataFrame(
            pca_coords, 
            columns=["PC1", "PC2"], 
            index=common_samples
        ).join(sub_meta)
        
        color_var = st.selectbox("Color by Metadata Column:", options=sub_meta.columns, index=0)
        
        var_exp = pca.explained_variance_ratio_ * 100
        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color=color_var,
            hover_name=pca_df.index,
            labels={
                "PC1": f"PC1 ({var_exp[0]:.1f}% Variance)",
                "PC2": f"PC2 ({var_exp[1]:.1f}% Variance)"
            },
            title="Sample PCA Projection"
        )
        fig_pca.update_traces(marker=dict(size=12))
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.warning("Not enough overlapping samples found between expression matrix columns and metadata rows.")

# --- TAB 3: VOLCANO PLOT ---
with tab_volcano:
    st.subheader("Differential Expression Analysis")
    
    cond_col = st.selectbox("Select Condition Column:", options=meta_df.columns, key="volcano_cond")
    unique_groups = meta_df[cond_col].dropna().unique()
    
    if len(unique_groups) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            grp_a = st.selectbox("Target Group (Treated):", options=unique_groups, index=0)
        with c2:
            grp_b = st.selectbox("Baseline Group (Control):", options=unique_groups, index=min(1, len(unique_groups)-1))
            
        if grp_a != grp_b:
            deg_df = compute_deg(expr_df, meta_df, condition_col=cond_col, group_a=grp_a, group_b=grp_b)
            
            fig_volcano = dashbio.VolcanoPlot(
                dataframe=deg_df,
                effect_size='EFFECTSIZE',
                p='P',
                gene='gene',
                effect_size_line=[-1.0, 1.0],
                genomewide_line_level=1.3,
                highlight_color="#EF553B",
                col="#636EFA"
            )
            fig_volcano.update_layout(title=f"{grp_a} vs {grp_b} Log2 Fold Change")
            st.plotly_chart(fig_volcano, use_container_width=True)
        else:
            st.warning("Please select two distinct groups to compare.")
    else:
        st.warning("Selected metadata column must contain at least two unique categories.")

# --- TAB 4: GENE INSPECTOR ---
with tab_gene:
    st.subheader("Individual Gene Expression Inspector")
    
    selected_gene = st.selectbox("Select Gene to Visualize:", options=expr_df.index)
    
    if selected_gene:
        gene_data = expr_df.loc[selected_gene].to_frame(name="Expression")
        gene_data = gene_data.join(meta_df, how="inner")
        
        group_var = st.selectbox("Group By:", options=meta_df.columns, key="gene_group")
        
        fig_gene = px.box(
            gene_data,
            x=group_var,
            y="Expression",
            points="all",
            color=group_var,
            title=f"Expression Profile for {selected_gene}",
            labels={"Expression": "Log2 Expression Level"}
        )
        st.plotly_chart(fig_gene, use_container_width=True)
