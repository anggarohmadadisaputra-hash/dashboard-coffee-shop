import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import random
from datetime import datetime, timedelta

# ==========================================
# 1. KONFIGURASI HALAMAN & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Brew Analytics - Coffee Shop Dashboard",
    page_icon="☕",
    layout="wide"
)

# Perbaikan CSS Kontras Warna & Desain UI Modern
st.markdown("""
    <style>
    /* Background Utama */
    .stApp {
        background-color: #FAF6F0;
        color: #2D2424;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #3E2723 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F5E6D3 !important;
    }
    
    /* Container/Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0D5C1 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03) !important;
    }
    
    /* Teks Metric Agar Terbaca Jelas */
    div[data-testid="stMetric"] label {
        color: #5D4037 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #2C1D11 !important;
        font-weight: 700 !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] button {
        color: #5D4037 !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #2C1D11 !important;
        border-bottom-color: #8D6E63 !important;
    }

    /* Recommendation Card Custom */
    .rec-card {
        background-color: #FFFFFF;
        border-left: 6px solid #8D6E63;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #2C1D11;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. GENERATOR DUMMY DATA DENGAN STRUKTUR REALISTIS
# ==========================================
@st.cache_data
def generate_dummy_data():
    np.random.seed(42)
    items_pool = [
        'Espresso', 'Cappuccino', 'Americano', 'Latte', 
        'Croissant', 'Waffle', 'Bagel', 'Choco Lava', 'Almond Pastry'
    ]
    data = []
    start_date = datetime(2026, 1, 1)
    
    for i in range(1, 501):
        trans_id = f"TRX-{1000+i}"
        date_val = start_date + timedelta(days=int(np.random.randint(0, 30)), hours=int(np.random.randint(7, 21)))
        
        rand_val = random.random()
        if rand_val < 0.4:
            basket = ['Cappuccino', 'Croissant']
        elif rand_val < 0.7:
            basket = ['Latte', 'Almond Pastry']
        else:
            basket = list(np.random.choice(items_pool, size=random.randint(1, 3), replace=False))
            
        for item in set(basket):
            data.append({
                'Transaction_ID': trans_id,
                'DateTime': date_val,
                'Item': item,
                'Price': np.random.choice([25000, 30000, 35000, 20000])
            })
            
    return pd.DataFrame(data)

# ==========================================
# 3. SIDEBAR: FILE UPLOADER & PARAMETER APRIORI
# ==========================================
st.sidebar.title("☕ BREW ANALYTICS")
st.sidebar.markdown("---")

# Menerima CSV dan Excel
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV / Excel)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("Dataset Berhasil Diunggah!")
    except Exception as e:
        st.sidebar.error("Gagal membaca file. Memakai data bawaan.")
        df = generate_dummy_data()
else:
    df = generate_dummy_data()

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parameter Apriori")

# Penyesuaian Dinamis Parameter Berdasarkan Ukuran Data
total_tx = df['Transaction_ID'].nunique() if 'Transaction_ID' in df.columns else len(df)
default_supp = 0.05 if total_tx < 1000 else 0.01

min_support = st.sidebar.slider("Minimum Support", 0.001, 0.500, float(default_supp), 0.005)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.05, 1.00, 0.30, 0.05)
min_lift = st.sidebar.slider("Minimum Lift Ratio", 0.5, 5.0, 1.0, 0.1)

# ==========================================
# 4. PREPROCESSING & ALGORITMA APRIORI
# ==========================================
if 'Transaction_ID' in df.columns and 'Item' in df.columns:
    basket_matrix = (df.groupby(['Transaction_ID', 'Item'])['Item']
                     .count().unstack().reset_index().fillna(0)
                     .set_index('Transaction_ID'))
    
    basket_sets = basket_matrix.map(lambda x: True if x >= 1 else False)
    
    # Eksekusi Apriori
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        rules = rules[rules['confidence'] >= min_confidence]
    else:
        rules = pd.DataFrame()
else:
    st.error("Kolom dataset harus memiliki nama: 'Transaction_ID' dan 'Item'.")
    rules = pd.DataFrame()

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================
st.title("📊 Analisis Pola Pembelian & Rekomendasi Menu")
st.caption("Coffee Shop Intelligence System - Apriori Algorithm")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Overview", 
    "🔗 Hasil Apriori", 
    "🎁 Rekomendasi Bundling", 
    "📥 Download Data"
])

# --- TAB 1: EXECUTIVE OVERVIEW ---
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    total_rev = df['Price'].sum() if 'Price' in df.columns else 0
    aov = total_rev / total_tx if total_tx > 0 else 0
    top_item = df['Item'].mode()[0] if 'Item' in df.columns else "-"
    
    c1.metric("Total Transaksi", f"{total_tx:,}")
    c2.metric("Total Omset", f"Rp {total_rev:,.0f}")
    c3.metric("Avg Order Value", f"Rp {aov:,.0f}")
    c4.metric("Menu Terlaris", top_item)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Top 10 Menu Paling Populer")
        item_counts = df['Item'].value_counts().reset_index()
        item_counts.columns = ['Menu', 'Jumlah']
        
        fig_bar = px.bar(item_counts.head(10), x='Jumlah', y='Menu', orientation='h',
                         color_discrete_sequence=['#8D6E63'])
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#2D2424", size=12),
            xaxis=dict(showgrid=True, gridcolor="#E0D5C1"),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_chart2:
        st.subheader("Tren Transaksi Harian")
        if 'DateTime' in df.columns:
            df['Date'] = pd.to_datetime(df['DateTime']).dt.date
            daily_trend = df.groupby('Date')['Transaction_ID'].nunique().reset_index()
            fig_line = px.line(daily_trend, x='Date', y='Transaction_ID',
                               color_discrete_sequence=['#3E2723'])
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#2D2424", size=12),
                yaxis=dict(showgrid=True, gridcolor="#E0D5C1")
            )
            st.plotly_chart(fig_line, use_container_width=True)

# --- TAB 2: HASIL APRIORI ---
with tab2:
    st.subheader("Frequent Itemsets & Rules")
    if rules.empty:
        st.warning("Aturan asosiasi tidak ditemukan. Silakan turunkan nilai Minimum Support atau Minimum Confidence pada sidebar.")
    else:
        st.write(f"Ditemukan **{len(rules)}** Aturan Asosiasi:")
        
        display_rules = rules.copy()
        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        st.dataframe(
            display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            .sort_values(by='lift', ascending=False),
            use_container_width=True
        )

# --- TAB 3: REKOMENDASI BUNDLING ---
with tab3:
    st.subheader("💡 Paket Bundling Menu Rekomendasi")
    if rules.empty:
        st.info("Sesuaikan parameter Apriori pada sidebar untuk menampilkan rekomendasi bundling.")
    else:
        top_rules = rules.sort_values(by='lift', ascending=False).head(6)
        cols = st.columns(2)
        idx = 0
        for _, row in top_rules.iterrows():
            ant = ', '.join(list(row['antecedents']))
            cons = ', '.join(list(row['consequents']))
            
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="rec-card">
                    <h4 style="margin:0; color:#3E2723;">☕ Paket Combo: {ant} + {cons}</h4>
                    <p style="margin:8px 0;">Pelanggan yang membeli <b>{ant}</b> berpeluang besar membeli <b>{cons}</b>.</p>
                    <small><b>Support:</b> {row['support']:.2f} | <b>Confidence:</b> {row['confidence']*100:.1f}% | <b>Lift:</b> {row['lift']:.2f}x</small>
                </div>
                """, unsafe_allow_html=True)
            idx += 1

# --- TAB 4: DOWNLOAD DATA ---
with tab4:
    st.subheader("📥 Unduh Laporan")
    c_dn1, c_dn2 = st.columns(2)
    with c_dn1:
        st.download_button(
            label="Download Data Transaksi (CSV)",
            data=df.to_csv(index=False),
            file_name="data_transaksi_coffee.csv",
            mime="text/csv"
        )
    with c_dn2:
        if not rules.empty:
            st.download_button(
                label="Download Aturan Asosiasi (CSV)",
                data=display_rules.to_csv(index=False),
                file_name="hasil_apriori_bundling.csv",
                mime="text/csv"
            )
