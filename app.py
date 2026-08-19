import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
import random
from datetime import datetime, timedelta

# ==========================================
# 1. KONFIGURASI HALAMAN & CUSTOM CSS (WARM LATTE TME)
# ==========================================
st.set_page_config(
    page_title="Brew Analytics - Apriori Dashboard",
    page_icon="☕",
    layout="wide"
)

# Custom CSS untuk warna, font, dan card styling
st.markdown("""
    <style>
    /* Background Utama */
    .stApp {
        background-color: #F8F4E1;
        color: #3E2723;
    }
    
    /* Container & Card Styling */
    div[data-testid="stMetric"], .css-1r6594q, .stTable {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #E0D5C1;
    }
    
    /* Custom Card Rekomendasi */
    .recommendation-card {
        background-color: #FFFFFF;
        border-left: 5px solid #C69C6D;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Header & Text */
    h1, h2, h3, h4 {
        color: #3E2723 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #AF8F6F;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. GENERATOR DUMMY DATA (JIKA TIDAK ADA UPLOAD)
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
    
    for i in range(1, 501):  # 500 Transaksi
        trans_id = f"TRX-{1000+i}"
        date_val = start_date + timedelta(days=int(np.random.randint(0, 30)), hours=int(np.random.randint(7, 21)))
        
        # Simulasi pola pembelian alami (Bundling Intentional)
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
# 3. SIDEBAR & FILTER
# ==========================================
st.sidebar.title("☕ BREW ANALYTICS")
st.sidebar.markdown("**Coffee Shop Pattern Analysis**")

uploaded_file = st.sidebar.file_uploader("Upload Dataset CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = generate_dummy_data()

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parameter Apriori")
min_support = st.sidebar.slider("Minimum Support", 0.01, 0.50, 0.05, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.10, 1.00, 0.30, 0.05)
min_lift = st.sidebar.slider("Minimum Lift Ratio", 1.0, 5.0, 1.2, 0.1)

# ==========================================
# 4. PREPROCESSING & ALGORITMA APRIORI
# ==========================================
# Format Data Matriks Transaksi (One-Hot Encoded)
basket_matrix = (df.groupby(['Transaction_ID', 'Item'])['Item']
                 .count().unstack().reset_index().fillna(0)
                 .set_index('Transaction_ID'))

def encode_units(x):
    return True if x >= 1 else False

basket_sets = basket_matrix.map(encode_units)

# Eksekusi Apriori
frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules = rules[rules['confidence'] >= min_confidence]
else:
    rules = pd.DataFrame()

# ==========================================
# 5. LAYOUT DASHBOARD (MULTI-TAB)
# ==========================================
st.title("📊 Analisis Pola Pembelian & Rekomendasi Bundling Menu")
st.caption("Aplikasi Analisis Association Rules Menggunakan Algoritma Apriori")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Overview", 
    "🔗 Hasil Apriori", 
    "🎁 Rekomendasi Bundling", 
    "📥 Download Data"
])

# --- TAB 1: EXECUTIVE OVERVIEW ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    total_tx = df['Transaction_ID'].nunique()
    total_rev = df['Price'].sum()
    aov = total_rev / total_tx if total_tx > 0 else 0
    top_item = df['Item'].mode()[0]
    
    col1.metric("Total Transaksi", f"{total_tx:,}")
    col2.metric("Total Omset", f"Rp {total_rev:,.0f}")
    col3.metric("Avg Order Value", f"Rp {aov:,.0f}")
    col4.metric("Menu Terlaris", top_item)
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 Menu Paling Populer")
        item_counts = df['Item'].value_counts().reset_index()
        item_counts.columns = ['Menu', 'Jumlah']
        fig_bar = px.bar(item_counts.head(10), x='Jumlah', y='Menu', orientation='h',
                         color_discrete_sequence=['#C69C6D'])
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.subheader("Tren Transaksi Harian")
        df['Date'] = pd.to_datetime(df['DateTime']).dt.date
        daily_trend = df.groupby('Date')['Transaction_ID'].nunique().reset_index()
        fig_line = px.line(daily_trend, x='Date', y='Transaction_ID',
                           color_discrete_sequence=['#00796B'])
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_line, use_container_width=True)

# --- TAB 2: HASIL ALGORITMA APRIORI ---
with tab2:
    st.subheader("Frequent Itemsets & Aturan Asosiasi")
    if rules.empty:
        st.warning("Tidak ada aturan asosiasi yang ditemukan dengan kombinasi parameter saat ini. Coba turunkan Nilai Support/Confidence.")
    else:
        st.write(f"Ditemukan **{len(rules)}** Aturan Asosiasi:")
        
        # Formatting tampilan aturan
        display_rules = rules.copy()
        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        st.dataframe(
            display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            .sort_values(by='lift', ascending=False),
            use_container_width=True
        )
        
        # Scatter Plot Rules
        st.subheader("Peta Aturan Asosiasi (Lift vs Confidence)")
        fig_scatter = px.scatter(
            display_rules, x='confidence', y='lift', size='support', color='lift',
            hover_data=['antecedents', 'consequents'],
            color_continuous_scale='Brwnyl'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 3: REKOMENDASI BUNDLING MENU ---
with tab3:
    st.subheader("💡 Paket Bundling Rekomendasi Sistem")
    if rules.empty:
        st.info("Ubah slider parameter di sidebar untuk menghasilkan paket bundling.")
    else:
        top_rules = rules.sort_values(by='lift', ascending=False).head(6)
        
        cols = st.columns(2)
        idx = 0
        for _, row in top_rules.iterrows():
            ant = ', '.join(list(row['antecedents']))
            cons = ', '.join(list(row['consequents']))
            
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>☕ Paket Combo: {ant} + {cons}</h4>
                    <p><b>Kombinasi:</b> Jika konsumen membeli <b>{ant}</b>, mereka memiliki kecenderungan tinggi untuk membeli <b>{cons}</b>.</p>
                    <small>Support: {row['support']:.2f} | Confidence: {row['confidence']*100:.1f}% | <b>Lift: {row['lift']:.2f}x</b></small>
                </div>
                """, unsafe_allow_html=True)
            idx += 1

# --- TAB 4: DOWNLOAD & LAPORAN ---
with tab4:
    st.subheader("📥 Unduh Laporan")
    st.write("Unduh dataset transaksi awal atau hasil aturan asosiasi dalam bentuk CSV:")
    
    c_dn1, c_dn2 = st.columns(2)
    with c_dn1:
        st.download_button(
            label="Download Raw Dataset CSV",
            data=df.to_csv(index=False),
            file_name="coffee_shop_transactions.csv",
            mime="text/csv"
        )
    with c_dn2:
        if not rules.empty:
            st.download_button(
                label="Download Association Rules CSV",
                data=display_rules.to_csv(index=False),
                file_name="apriori_bundling_rules.csv",
                mime="text/csv"
            )
