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

st.markdown('''
    <style>
    .stApp { background-color: #FAF6F0; color: #2D2424; }
    section[data-testid="stSidebar"] { background-color: #3E2723 !important; }
    section[data-testid="stSidebar"] * { color: #F5E6D3 !important; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0D5C1 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03) !important;
    }
    div[data-testid="stMetric"] label { color: #5D4037 !important; font-weight: 600 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #2C1D11 !important; font-weight: 700 !important; }
    .rec-card {
        background-color: #FFFFFF;
        border-left: 6px solid #8D6E63;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        color: #2C1D11;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
''', unsafe_allow_html=True)

# ==========================================
# 2. GENERATOR DUMMY DATA
# ==========================================
@st.cache_data
def generate_dummy_data():
    np.random.seed(42)
    items_pool = ['Espresso', 'Cappuccino', 'Americano', 'Latte', 'Croissant', 'Waffle', 'Bagel', 'Choco Lava', 'Almond Pastry']
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
            data.append({'Transaction_ID': trans_id, 'DateTime': date_val, 'Item': item, 'Price': np.random.choice([25000, 30000, 35000, 20000])})
    return pd.DataFrame(data)

# ==========================================
# 3. FUNGSI MAPPER KOLOM OTOMATIS
# ==========================================
def map_columns_automatically(df_raw):
    col_mapping = {}
    cols = df_raw.columns.tolist()
    
    # Deteksi Kolom Transaksi ID
    for c in cols:
        if any(k in c.lower() for k in ['trans', 'id', 'order', 'nota', 'receipt', 'faktur']):
            col_mapping[c] = 'Transaction_ID'
            break
            
    # Deteksi Kolom Nama Item/Menu
    for c in cols:
        if any(k in c.lower() for k in ['item', 'product', 'menu', 'barang', 'coffee', 'nama']):
            col_mapping[c] = 'Item'
            break
            
    # Deteksi Kolom Harga/Nominal
    for c in cols:
        if any(k in c.lower() for k in ['price', 'harga', 'amount', 'total', 'grand', 'bayar']):
            col_mapping[c] = 'Price'
            break
            
    # Deteksi Kolom Waktu/Tanggal
    for c in cols:
        if any(k in c.lower() for k in ['date', 'time', 'tgl', 'tanggal', 'waktu']):
            col_mapping[c] = 'DateTime'
            break
            
    return df_raw.rename(columns=col_mapping)

# ==========================================
# 4. SIDEBAR & FILE LOADING
# ==========================================
st.sidebar.title("☕ BREW ANALYTICS")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV / Excel)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
        
        df = map_columns_automatically(df_input)
        st.sidebar.success("Dataset Berhasil Dimuat!")
    except Exception as e:
        st.sidebar.error("Format file tidak terbaca. Menggunakan Dummy Data.")
        df = generate_dummy_data()
else:
    df = generate_dummy_data()

# Penanganan fallback jika kolom dasar masih belum ada
if 'Transaction_ID' not in df.columns:
    df['Transaction_ID'] = [f"TRX-{i//2 + 1}" for i in range(len(df))]
if 'Item' not in df.columns:
    df['Item'] = df.iloc[:, 0].astype(str)
if 'Price' not in df.columns:
    df['Price'] = 25000

# ==========================================
# 5. PARAMETER APRIORI DINAMIS DAN OTOMATIS
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parameter Apriori")

total_transactions = df['Transaction_ID'].nunique()

# Formula otomatis menghitung nilai Support optimal dari ukuran data
if total_transactions > 2000:
    auto_supp = 0.005
elif total_transactions > 500:
    auto_supp = 0.01
else:
    auto_supp = 0.03

min_support = st.sidebar.slider("Minimum Support", 0.001, 0.300, float(auto_supp), 0.002)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.05, 1.00, 0.20, 0.05)
min_lift = st.sidebar.slider("Minimum Lift Ratio", 0.1, 5.0, 1.0, 0.1)

# ==========================================
# 6. PEMROSESAN ALGORITMA APRIORI
# ==========================================
basket_matrix = (df.groupby(['Transaction_ID', 'Item'])['Item']
                 .count().unstack().reset_index().fillna(0)
                 .set_index('Transaction_ID'))

basket_sets = basket_matrix.map(lambda x: True if x >= 1 else False)

frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    if not rules.empty:
        rules = rules[rules['confidence'] >= min_confidence]
else:
    rules = pd.DataFrame()

# ==========================================
# 7. LAYOUT UTAMA DASHBOARD
# ==========================================
st.title("📊 Dashboard Analisis Apriori Coffee Shop")
st.caption("Sistem Rekomendasi Bundling Otomatis Berbasis Data Transaksi")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Executive Overview", "🔗 Hasil Apriori", "🎁 Rekomendasi Bundling", "📥 Download Data"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    total_rev = df['Price'].sum() if 'Price' in df.columns else 0
    aov = total_rev / total_transactions if total_transactions > 0 else 0
    top_item = df['Item'].mode()[0] if not df['Item'].empty else "-"
    
    c1.metric("Total Transaksi", f"{total_transactions:,}")
    c2.metric("Total Omset", f"Rp {total_rev:,.0f}")
    c3.metric("Avg Order Value", f"Rp {aov:,.0f}")
    c4.metric("Menu Terlaris", top_item)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Menu Paling Populer")
        item_counts = df['Item'].value_counts().reset_index()
        item_counts.columns = ['Menu', 'Jumlah']
        fig_bar = px.bar(item_counts.head(10), x='Jumlah', y='Menu', orientation='h', color_discrete_sequence=['#8D6E63'])
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#2D2424", size=12),
            xaxis=dict(showgrid=True, gridcolor="#E0D5C1"),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        st.subheader("Tren Transaksi Harian")
        if 'DateTime' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['DateTime']).dt.date
                daily = df.groupby('Date')['Transaction_ID'].nunique().reset_index()
                fig_line = px.line(daily, x='Date', y='Transaction_ID', color_discrete_sequence=['#3E2723'])
                fig_line.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2D2424", size=12),
                    yaxis=dict(showgrid=True, gridcolor="#E0D5C1")
                )
                st.plotly_chart(fig_line, use_container_width=True)
            except:
                st.info("Format kolom tanggal tidak valid untuk grafik tren.")
        else:
            st.info("Kolom tanggal tidak ditemukan di dataset.")

with tab2:
    st.subheader("Frequent Itemsets & Association Rules")
    if not rules.empty:
        display_rules = rules.copy()
        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        st.dataframe(display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False), use_container_width=True)
    else:
        st.warning("Aturan asosiasi tidak ditemukan. Silakan turunkan Minimum Support / Confidence pada sidebar.")

with tab3:
    st.subheader("💡 Paket Bundling Rekomendasi Sistem")
    if not rules.empty:
        top_rules = rules.sort_values(by='lift', ascending=False).head(6)
        cols = st.columns(2)
        idx = 0
        for _, row in top_rules.iterrows():
            ant = ', '.join(list(row['antecedents']))
            cons = ', '.join(list(row['consequents']))
            with cols[idx % 2]:
                st.markdown(f'''
                <div class="rec-card">
                    <h4 style="margin:0; color:#3E2723;">☕ Paket Combo: {ant} + {cons}</h4>
                    <p style="margin:8px 0;">Konsumen yang membeli <b>{ant}</b> memiliki potensi tinggi membeli <b>{cons}</b>.</p>
                    <small><b>Support:</b> {row['support']:.3f} | <b>Confidence:</b> {row['confidence']*100:.1f}% | <b>Lift Ratio: {row['lift']:.2f}x</b></small>
                </div>
                ''', unsafe_allow_html=True)
            idx += 1
    else:
        st.info("Tidak ada kombinasi bundling yang terbentuk pada tingkat Support/Confidence saat ini.")

with tab4:
    st.subheader("📥 Download Data")
    st.download_button("Download Processed Data (CSV)", df.to_csv(index=False), "processed_transactions.csv", "text/csv")
