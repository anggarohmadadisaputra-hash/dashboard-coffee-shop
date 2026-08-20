import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import re
from datetime import datetime

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA COFFEE SHOP
# ==========================================
st.set_page_config(
    page_title="Kelompok 8 Analytics - Coffee Shop Dashboard",
    page_icon="☕",
    layout="wide"
)

st.markdown('''
    <style>
    /* Background Utama */
    .stApp {
        background-color: #FAF6F0;
        color: #2D2424;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #2C1D11 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F5E6D3 !important;
    }
    
    /* Cards & Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0D5C1 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04) !important;
    }
    div[data-testid="stMetric"] label {
        color: #5D4037 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #2C1D11 !important;
        font-weight: 800 !important;
    }

    /* Custom Recommendation Card */
    .bundle-card {
        background-color: #FFFFFF;
        border-left: 6px solid #8D6E63;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        color: #2D2424;
    }
    .bundle-card h4 {
        color: #3E2723;
        margin-top: 0;
        margin-bottom: 8px;
        font-weight: 700;
    }
    .bundle-badge {
        background-color: #D7CCC8;
        color: #3E2723;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 6px;
    }
    
    /* Tab Header Custom */
    button[data-baseweb="tab"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #5D4037 !important;
    }
    button[aria-selected="true"] {
        color: #3E2723 !important;
        border-bottom-color: #8D6E63 !important;
    }
    </style>
''', unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI PARSER UNTUK COFFEE_SHOP_SALES
# ==========================================
def parse_coffee_sales_data(uploaded_file):
    try:
        # Coba baca file dengan delimiter titik koma (;) atau koma (,)
        try:
            df_raw = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
            if len(df_raw.columns) <= 1:
                uploaded_file.seek(0)
                df_raw = pd.read_csv(uploaded_file, sep=',', encoding='utf-8-sig')
        except Exception:
            uploaded_file.seek(0)
            df_raw = pd.read_excel(uploaded_file)

        rows = []
        for idx, row in df_raw.iterrows():
            tx_id = row.get('transaction_id', idx)
            date_str = str(row.get('transaction_date', ''))
            time_str = str(row.get('transaction_time', ''))
            dt_combined = f"{date_str} {time_str}".strip()

            # Deteksi kolom produk & harga
            product_str = str(row.get('product_detail', row.get('product_type', row.get('Item', ''))))
            price_str = str(row.get('unit_price', row.get('Price', '0')))

            # Split berdasarkan separator Pipe ('|')
            items = [i.strip() for i in product_str.split('|')]
            prices = [p.strip().replace(',', '.') for p in price_str.split('|')]

            for i, item_name in enumerate(items):
                if not item_name or item_name.lower() == 'nan':
                    continue
                try:
                    p_val = float(re.sub(r'[^0-9.]', '', prices[i])) if i < len(prices) else 0.0
                except Exception:
                    p_val = 0.0

                rows.append({
                    'Transaction_ID': tx_id,
                    'DateTime': dt_combined,
                    'Item': item_name,
                    'Price': p_val
                })

        return pd.DataFrame(rows)
    except Exception as e:
        st.sidebar.error(f"Error membaca file: {e}")
        return pd.DataFrame()

# ==========================================
# 3. SIDEBAR & FILE LOADING
# ==========================================
st.sidebar.title("☕ Coffe Shop ANALYTICS")
st.sidebar.caption("Coffee Shop Market Basket Analysis")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Dataset Coffee Shop (CSV/Excel)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    df = parse_coffee_sales_data(uploaded_file)
    if not df.empty:
        st.sidebar.success("Dataset Coffee Shop Berhasil Dimuat!")
    else:
        st.sidebar.error("Dataset gagal diproses.")
else:
    st.info("👋 Silakan unggah file **Coffee_Shop_Sales.csv** Anda di sidebar kiri untuk memulai analisis.")
    st.stop()

# ==========================================
# 4. PARAMETER APRIORI
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parameter Apriori")

total_tx = df['Transaction_ID'].nunique()
auto_supp = 0.015 if total_tx > 1000 else 0.03

min_support = st.sidebar.slider("Minimum Support", 0.001, 0.200, float(auto_supp), 0.002)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.05, 1.00, 0.20, 0.05)
min_lift = st.sidebar.slider("Minimum Lift Ratio", 0.5, 5.0, 1.0, 0.1)

# ==========================================
# 5. PEMROSESAN ALGORITMA APRIORI
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
# 6. DASHBOARD MAIN LAYOUT
# ==========================================
st.title("🛒 Market Basket Analysis & Rekomendasi Bundling")
st.markdown("Aplikasi interaktif untuk menemukan pola kombinasi pembelian menu **Coffee Shop** menggunakan **Algoritma Apriori**.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Ringkasan Data (EDA)", "📋 Association Rules", "💡 Simulator Bundling", "🕸️ Network Graph"])

# --- TAB 1: EDA ---
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    total_rev = float(df['Price'].sum())
    aov = total_rev / total_tx if total_tx > 0 else 0
    top_item = str(df['Item'].mode()[0]) if not df['Item'].empty else "-"

    m1.metric("Total Struk Transaksi", f"{total_tx:,}")
    m2.metric("Total Item Terjual", f"{len(df):,}")
    m3.metric("Estimasi Total Omset", f"$ {total_rev:,.2f}")
    m4.metric("Menu Terlaris", top_item)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("☕ Top 10 Menu Paling Populer")
        item_counts = df['Item'].value_counts().reset_index()
        item_counts.columns = ['Menu', 'Jumlah']
        fig_bar = px.bar(item_counts.head(10), x='Jumlah', y='Menu', orientation='h',
                         color_discrete_sequence=['#8D6E63'])
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#2D2424", size=12),
            xaxis=dict(showgrid=True, gridcolor="#E0D5C1"),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("📅 Tren Transaksi Harian")
        try:
            df['Date'] = pd.to_datetime(df['DateTime'], errors='coerce').dt.date
            daily = df.groupby('Date')['Transaction_ID'].nunique().reset_index()
            fig_line = px.line(daily, x='Date', y='Transaction_ID', color_discrete_sequence=['#3E2723'])
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#2D2424", size=12),
                yaxis=dict(showgrid=True, gridcolor="#E0D5C1")
            )
            st.plotly_chart(fig_line, use_container_width=True)
        except Exception:
            st.info("Gagal mengurai tanggal transaksi untuk grafik harian.")

# --- TAB 2: ASSOCIATION RULES ---
with tab2:
    st.subheader("📋 Tabel Association Rules")
    if not rules.empty:
        display_rules = rules.copy()
        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        st.dataframe(display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False), use_container_width=True)
    else:
        st.warning("Tidak ada aturan (rules) yang terbentuk. Silakan turunkan Minimum Support atau Confidence di Sidebar.")

# --- TAB 3: SIMULATOR BUNDLING ---
with tab3:
    st.subheader("💡 Simulator Rekomendasi Bundling")
    if not rules.empty:
        all_antecedents = sorted(list(set([item for sublist in rules['antecedents'] for item in sublist])))
        selected_item = st.selectbox("Pilih Produk Utama Pelanggan:", all_antecedents)

        recommendations = rules[rules['antecedents'].apply(lambda x: selected_item in list(x))].sort_values(by='lift', ascending=False)

        if not recommendations.empty:
            st.success(f"Ditemukan rekomendasi paket bundling untuk **{selected_item}**!")
            cols = st.columns(2)
            idx = 0
            for _, row in recommendations.iterrows():
                cons = ', '.join(list(row['consequents']))
                with cols[idx % 2]:
                    st.markdown(f'''
                    <div class="bundle-card">
                        <h4>☕ Paket Combo: {selected_item} + {cons}</h4>
                        <p style="margin: 6px 0;">Pelanggan yang membeli <b>{selected_item}</b> cenderung berpotensi membeli <b>{cons}</b>.</p>
                        <span class="bundle-badge">Support: {row['support']:.3f}</span>
                        <span class="bundle-badge">Confidence: {row['confidence']*100:.1f}%</span> 
                        <span class="bundle-badge">Lift: {row['lift']:.2f}x</span>
                    </div>
                    ''', unsafe_allow_html=True)
                idx += 1
        else:
            st.info(f"Belum ada rekomendasi menu pendamping untuk **{selected_item}** pada nilai parameter saat ini.")
    else:
        st.info("Aturan asosiasi belum terbentuk. Sesuaikan slider Support/Confidence terlebih dahulu.")

# --- TAB 4: NETWORK GRAPH ---
with tab4:
    st.subheader("🕸️ Visualisasi Jaringan Hubungan Produk")
    if not rules.empty and len(rules) > 0:
        G = nx.DiGraph()
        for i in range(len(rules)):
            ant = ', '.join(list(rules.iloc[i]['antecedents']))
            con = ', '.join(list(rules.iloc[i]['consequents']))
            G.add_edge(ant, con, weight=rules.iloc[i]['lift'])

        pos = nx.spring_layout(G, k=0.8, seed=42)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#8D6E63'), hoverinfo='none', mode='lines')

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=node_text,
            textposition="top center", hoverinfo='text',
            marker=dict(size=22, color='#3E2723', line=dict(width=2, color='#D7CCC8'))
        )

        fig_net = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               showlegend=False, hovermode='closest',
                               margin=dict(b=0, l=0, r=0, t=0),
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                           ))
        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.warning("Tidak dapat menampilkan grafik jaringan karena belum ada aturan asosiasi yang terbentuk.")
