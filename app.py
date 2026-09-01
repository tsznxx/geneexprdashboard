import io
import numpy as np
import pandas as pd
import plotly.express as px
import dash_bio as dashbio
import streamlit as st
from sklearn.decomposition import PCA

# Configure page layout
st.set_page_config(
    page_title="Gene Expression Dashboard",
    page_icon="🧬",
    layout="wide"
)

# ---------------------------------------------------------
# HELPER FUNCTIONS & DEMO DATA GENERATORS
# ---------------------------------------------------------
def get_demo_data():
    """Generates synthetic expression matrix and metadata."""
    np.random.seed(42)
    genes = [f"Gene_{i}" for i in range(1, 101)]
    # Add a few recognized genes for display
    genes[:5] = ["BRCA1", "TP53", "EGFR", "MYC", "TNF"]
    
    samples_ctrl = [f"Control_{i}" for i in range(1, 6)]
    samples_treat = [f"Treated_{i}" for i in range(1, 6)]
    all_samples = samples_ctrl + samples_treat

    # Generate log2 expression data
    data_ctrl = np.random.normal(loc=5.0, scale=1.5, size=(100, 5))
    data_treat = np.random.normal(loc=5.5, scale=1.5, size=(100, 5))
    
    # Introduce differential expression in top 10 genes
    data_treat[:10, :] += np.random.uniform(2.0, 4.0, size=(10, 5))
    
    expr_df = pd.DataFrame(
        np.hstack([data_ctrl, data_treat]), 
        index=genes, 
        columns=all_samples
    )
    
    meta_df = pd.DataFrame({
        "Condition": ["Control"] * 5 + ["Treated"] * 5,
        "Batch": ["Batch_1", "Batch_2", "Batch_1", "Batch_2", "Batch_1"] * 2
    }, index=all_samples)
    
    return expr_df, meta_df

def compute_deg(expr_df, meta_df, condition_col="Condition", group_a="Treated", group_b="Control"):
    """Simple t-test wrapper to calculate Fold Change and p-values for Volcano plot."""
    from scipy import stats
    
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
    
    # Section 1: Load Example Data
    st.subheader("1. Demo Data")
    if st.button("⚡ Load Example Datasets", use_container_width=True):
        demo_expr, demo_meta = get_demo_data()
        st.session_state["expr_df"] = demo_expr
        st.session_state["meta_df"] = demo_meta
        st.success("Loaded synthetic 100-gene dataset!")

    st.divider()

    # Section 2: Upload Custom Data
    st.subheader("2. Upload Your Data")
    
    uploaded_expr = st.file_uploader("Upload Expression Matrix", type=["csv", "tsv", "txt"], key="up_expr")
    if uploaded_expr:
        sep = "\t" if uploaded_expr.name.endswith((".tsv", ".txt")) else ","
        st.session_state["expr_df"] = pd.read_csv(uploaded_expr, index_col=0, sep=sep)
        st.caption(f"Loaded: `{uploaded_expr.name}`")

    uploaded_meta = st.file_uploader("Upload Metadata", type=["csv", "tsv", "txt"], key="up_meta")
    if uploaded_meta:
        sep = "\t" if uploaded_meta.name.endswith((".tsv", ".txt")) else ","
        st.session_state["meta_df"] = pd.read_csv(uploaded_meta, index_col=0, sep=sep)
        st.caption(f"Loaded: `{uploaded_meta.name}`")

    st.divider()

    # Section 3: Downloads
    st.subheader("3. Downloads")
    demo_expr, demo_meta = get_demo_data()
    
    st.download_button(
        label="📥 Download Example Expression CSV",
        data=demo_expr.to_csv().encode('utf-8'),
        file_name="example_expression.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.download_button(
        label="📥 Download Example Metadata CSV",
        data=demo_meta.to_csv().encode('utf-8'),
        file_name="example_metadata.csv",
        mime="text/csv",
        use_container_width=True
    )


# ---------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# ---------------------------------------------------------
st.title("Gene Expression Dashboard")

# Check if data exists in session state
if "expr_df" not in st.session_state or "meta_df" not in st.session_state:
    st.info("👈 Please load the example datasets or upload your own files via the sidebar to begin.")
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
        st.subheader("Expression Matrix Preview")
        st.caption(f"Shape: {expr_df.shape[0]} Genes × {expr_df.shape[1]} Samples")
        st.dataframe(expr_df.head(10), use_container_width=True)
    with col2:
        st.subheader("Metadata Preview")
        st.caption(f"Shape: {meta_df.shape[0]} Samples × {meta_df.shape[1]} Attributes")
        st.dataframe(meta_df.head(10), use_container_width=True)

# --- TAB 2: PCA ANALYSIS ---
with tab_pca:
    st.subheader("Principal Component Analysis (PCA)")
    
    # Perform PCA on transposed expression matrix (Samples as rows)
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(expr_df.T)
    
    pca_df = pd.DataFrame(
        pca_coords, 
        columns=["PC1", "PC2"], 
        index=expr_df.columns
    )
    
    # Merge with sample metadata
    pca_df = pca_df.join(meta_df)
    
    # Color/shape selector
    color_var = st.selectbox("Color by Metadata Column:", options=meta_df.columns, index=0)
    
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

# --- TAB 3: VOLCANO PLOT ---
with tab_volcano:
    st.subheader("Differential Expression Analysis")
    
    # Select conditions to compare
    cond_col = st.selectbox("Select Condition Column:", options=meta_df.columns, key="volcano_cond")
    unique_groups = meta_df[cond_col].unique()
    
    if len(unique_groups) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            grp_a = st.selectbox("Target Group (Treated):", options=unique_groups, index=0)
        with c2:
            grp_b = st.selectbox("Baseline Group (Control):", options=unique_groups, index=min(1, len(unique_groups)-1))
            
        if grp_a != grp_b:
            deg_df = compute_deg(expr_df, meta_df, condition_col=cond_col, group_a=grp_a, group_b=grp_b)
            
            # Interactive Dash-Bio Volcano Plot
            fig_volcano = dashbio.VolcanoPlot(
                dataframe=deg_df,
                effect_size='EFFECTSIZE',
                p='P',
                gene='gene',
                effect_size_line=[-1.0, 1.0],
                genomewide_line_level=1.3,  # equivalent to p=0.05
                highlight_color="#EF553B",
                col="#636EFA"
            )
            fig_volcano.update_layout(title=f"{grp_a} vs {grp_b} Log2 Fold Change")
            st.plotly_chart(fig_volcano, use_container_width=True)
        else:
            st.warning("Please select two distinct groups to compare.")
    else:
        st.warning("Selected metadata column must contain at least two groups.")

# --- TAB 4: GENE INSPECTOR ---
with tab_gene:
    st.subheader("Individual Gene Expression Inspector")
    
    selected_gene = st.selectbox("Select Gene to Visualize:", options=expr_df.index)
    
    if selected_gene:
        gene_data = expr_df.loc[selected_gene].to_frame(name="Expression")
        gene_data = gene_data.join(meta_df)
        
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
