import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import random
from datetime import datetime, timedelta
import io

# --- 1. SETTINGS & CSS ---
st.set_page_config(page_title="Coffee Shop Apriori Dashboard", page_icon="☕", layout="wide")

# Custom CSS for Coffee Shop Vibe
st.markdown("""
<style>
    :root {
        --espresso: #3B2F2F;
        --warm-cream: #F5E6D3;
        --latte-gold: #D4A373;
        --accent-brown: #6F4E37;
    }
    .stApp {
        background-color: var(--warm-cream);
        color: var(--espresso);
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--espresso) !important;
        font-family: 'Georgia', serif;
    }
    .css-1d391kg {  /* Sidebar */
        background-color: var(--espresso) !important;
    }
    .css-1d391kg * {
        color: var(--warm-cream) !important;
    }
    .stButton>button {
        background-color: var(--latte-gold);
        color: var(--espresso);
        border: None;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: var(--accent-brown);
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid var(--latte-gold);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DUMMY DATA GENERATOR ---
@st.cache_data
def generate_dummy_data(n_transactions=1000):
    menu_items = ['Espresso', 'Cappuccino', 'Americano', 'Latte', 'Mocha', 'Croissant', 'Waffle', 'Bagel', 'Choco Lava', 'Muffin', 'Sandwich', 'Tea']
    
    data = []
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(n_transactions):
        t_date = start_date + timedelta(days=random.randint(0, 30))
        
        # Determine time of day and adjust probabilities
        hour = random.randint(7, 22)
        if 7 <= hour < 11:
            time_category = 'Pagi'
            weights = [0.15, 0.15, 0.2, 0.1, 0.05, 0.15, 0.05, 0.1, 0.0, 0.05, 0.0, 0.0]
        elif 11 <= hour < 15:
            time_category = 'Siang'
            weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.2, 0.1]
        elif 15 <= hour < 19:
            time_category = 'Sore'
            weights = [0.05, 0.15, 0.05, 0.15, 0.15, 0.1, 0.1, 0.05, 0.1, 0.05, 0.0, 0.05]
        else:
            time_category = 'Malam'
            weights = [0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.2, 0.0, 0.2, 0.1, 0.0, 0.1]
            
        n_items = random.choices([1, 2, 3, 4, 5], weights=[0.3, 0.4, 0.2, 0.08, 0.02])[0]
        items_bought = list(set(random.choices(menu_items, weights=weights, k=n_items)))
        
        data.append({
            'Transaction_ID': f"TXN_{i:04d}",
            'Date': t_date.strftime('%Y-%m-%d'),
            'Time': f"{hour:02d}:{random.randint(0,59):02d}",
            'Time_Category': time_category,
            'Items_Purchased': ", ".join(items_bought)
        })
        
    return pd.DataFrame(data)

# --- 3. HELPER FUNCTIONS ---
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def prepare_apriori_data(df):
    # Split the items and explode
    df['Items_List'] = df['Items_Purchased'].str.split(', ')
    df_exploded = df.explode('Items_List')
    df_exploded['Quantity'] = 1
    
    # Create basket
    basket = (df_exploded
          .groupby(['Transaction_ID', 'Items_List'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Transaction_ID'))
    
    basket_sets = basket.map(encode_units)
    return basket_sets

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3218/3218047.png", width=100)
    st.title("☕ Coffee Analytics")
    st.subheader("Pengaturan Data & Model")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            st.success("File berhasil diupload!")
            
            st.markdown("---")
            st.subheader("Pemetaan Kolom (Data Mapping)")
            st.info("Pilih kolom dari dataset Anda yang sesuai:")
            
            col_names = raw_df.columns.tolist()
            
            # Helper to find default index
            def find_idx(keywords):
                for i, col in enumerate(col_names):
                    if any(k in str(col).lower() for k in keywords):
                        return i
                return 0
                
            tx_col = st.selectbox("Kolom ID Transaksi:", col_names, index=find_idx(['id', 'trans', 'nota']))
            date_col = st.selectbox("Kolom Tanggal:", col_names, index=find_idx(['date', 'tanggal', 'waktu']))
            item_col = st.selectbox("Kolom Nama Produk/Menu:", col_names, index=find_idx(['item', 'product', 'menu', 'barang']))
            
            # Process data
            raw_df = raw_df.dropna(subset=[tx_col, date_col, item_col])
            
            # Group by transaction ID to combine items
            grouped = raw_df.groupby([tx_col, date_col])[item_col].apply(lambda x: ', '.join(x.astype(str))).reset_index()
            grouped.columns = ['Transaction_ID', 'Date', 'Items_Purchased']
            
            # Try to get time if available
            time_col = st.selectbox("Kolom Jam (Opsional):", ["Kosongkan"] + col_names, index=0)
            if time_col != "Kosongkan":
                time_grouped = raw_df.groupby([tx_col, date_col])[time_col].first().reset_index()
                grouped['Time'] = time_grouped[time_col]
            else:
                grouped['Time'] = "00:00"
                
            df = grouped
            
        except Exception as e:
            st.error(f"Error memproses file: {e}")
            df = generate_dummy_data()
    else:
        st.info("Menggunakan Dummy Data. Silakan upload file untuk analisis data Anda sendiri.")
        df = generate_dummy_data()

    st.markdown("---")
    st.subheader("Parameter Apriori")
    min_support = st.slider("Minimum Support", min_value=0.01, max_value=0.50, value=0.05, step=0.01)
    min_confidence = st.slider("Minimum Confidence", min_value=0.10, max_value=1.00, value=0.30, step=0.05)
    min_lift = st.slider("Minimum Lift", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
    
    st.markdown("---")
    st.subheader("Filter Data")
    
    # Ensure Date is datetime
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
    except:
        pass
    
    if not df.empty:
        date_range = st.date_input(
            "Rentang Tanggal",
            value=(df['Date'].min(), df['Date'].max()),
            min_value=df['Date'].min(),
            max_value=df['Date'].max()
        )
    else:
        date_range = []
    
    if 'Time_Category' in df.columns:
        time_filter = st.multiselect(
            "Waktu Transaksi",
            options=df['Time_Category'].unique(),
            default=df['Time_Category'].unique()
        )
    else:
        time_filter = []

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
    filtered_df = df.loc[mask]
else:
    filtered_df = df

if time_filter and 'Time_Category' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Time_Category'].isin(time_filter)]

# --- MAIN DASHBOARD ---
st.title("☕ Analisis Pola Pembelian Konsumen")
st.markdown("*Rekomendasi Bundling Menu Coffee Shop dengan Algoritma Apriori*")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview & EDA", 
    "🧬 Algoritma Apriori & Itemsets", 
    "💡 Rekomendasi Bundling", 
    "📥 Unduh Laporan"
])

# Process Apriori early as it's needed in multiple tabs
basket_sets = prepare_apriori_data(filtered_df)
# using use_colnames=True to show the item names instead of column indices
frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
rules = pd.DataFrame()
if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules = rules[rules['lift'] >= min_lift]
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# --- TAB 1: EDA ---
with tab1:
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_tx = len(filtered_df)
    items_list = filtered_df['Items_Purchased'].str.split(', ').explode()
    total_items = len(items_list)
    avg_basket = total_items / total_tx if total_tx > 0 else 0
    top_product = items_list.value_counts().index[0] if total_items > 0 else "-"
    
    col1.markdown(f"""
    <div class="metric-card">
        <h3 style="color: var(--accent-brown); margin:0;">Total Transaksi</h3>
        <h2 style="margin:0;">{total_tx}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col2.markdown(f"""
    <div class="metric-card">
        <h3 style="color: var(--accent-brown); margin:0;">Item Terjual</h3>
        <h2 style="margin:0;">{total_items}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col3.markdown(f"""
    <div class="metric-card">
        <h3 style="color: var(--accent-brown); margin:0;">Avg Basket Size</h3>
        <h2 style="margin:0;">{avg_basket:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col4.markdown(f"""
    <div class="metric-card">
        <h3 style="color: var(--accent-brown); margin:0;">Produk Terlaris</h3>
        <h2 style="margin:0;">{top_product}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("🏆 Top 10 Produk Terlaris")
        top_10 = items_list.value_counts().head(10).reset_index()
        top_10.columns = ['Menu', 'Jumlah']
        fig1 = px.bar(top_10, x='Jumlah', y='Menu', orientation='h', 
                      color='Jumlah', color_continuous_scale=['#D4A373', '#3B2F2F'])
        fig1.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_chart2:
        st.subheader("🕒 Tren Transaksi Berdasarkan Jam")
        if not filtered_df.empty and 'Time' in filtered_df.columns:
            try:
                # Convert time to string, handle mixed formats
                filtered_df['Hour'] = pd.to_datetime(filtered_df['Time'].astype(str), format='mixed', errors='coerce').dt.hour
                hourly_tx = filtered_df.dropna(subset=['Hour']).groupby('Hour').size().reset_index(name='Jumlah Transaksi')
                if not hourly_tx.empty:
                    fig2 = px.line(hourly_tx, x='Hour', y='Jumlah Transaksi', markers=True, 
                                   line_shape='spline', color_discrete_sequence=['#6F4E37'])
                    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Format waktu pada kolom tidak dapat diproses.")
            except Exception as e:
                st.info("Tidak dapat menampilkan grafik jam.")
        else:
            st.info("Data jam tidak tersedia.")
        
    st.subheader("🛒 Distribusi Ukuran Keranjang (Basket Size)")
    if not filtered_df.empty:
        basket_sizes = filtered_df['Items_Purchased'].str.split(', ').apply(len).value_counts().reset_index()
        basket_sizes.columns = ['Jumlah Item per Transaksi', 'Frekuensi']
        basket_sizes = basket_sizes.sort_values('Jumlah Item per Transaksi')
        fig3 = px.bar(basket_sizes, x='Jumlah Item per Transaksi', y='Frekuensi',
                      color_discrete_sequence=['#D4A373'])
        fig3.update_layout(xaxis=dict(tickmode='linear'), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)


# --- TAB 2: APRIORI ---
with tab2:
    st.subheader("🧬 Frequent Itemsets & Association Rules")
    
    if frequent_itemsets.empty:
        st.warning("Tidak ada itemset yang memenuhi nilai Minimum Support. Silakan turunkan nilai Minimum Support di sidebar.")
    elif rules.empty:
        st.warning("Tidak ada rules yang memenuhi kriteria Confidence dan Lift. Silakan turunkan nilainya di sidebar.")
    else:
        col_rule1, col_rule2 = st.columns(2)
        
        with col_rule1:
            st.write(f"**Frequent Itemsets (Support >= {min_support})**")
            frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
            st.dataframe(frequent_itemsets[['itemsets_str', 'support']].sort_values('support', ascending=False), use_container_width=True)
            
        with col_rule2:
            st.write(f"**Association Rules (Conf >= {min_confidence}, Lift >= {min_lift})**")
            display_rules = rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].copy()
            display_rules.columns = ['Jika Membeli (Antecedent)', 'Maka Membeli (Consequent)', 'Support', 'Confidence', 'Lift']
            st.dataframe(display_rules.sort_values('lift', ascending=False), use_container_width=True)

        st.subheader("🕸️ Network Graph Association Rules")
        
        # Build Network Graph
        G = nx.DiGraph()
        for i, row in rules.iterrows():
            G.add_edge(row['antecedents_str'], row['consequents_str'], weight=row['lift'])
            
        pos = nx.spring_layout(G, k=0.5)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#6F4E37'),
            hoverinfo='none',
            mode='lines')
            
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="bottom center",
            hoverinfo='text',
            marker=dict(
                color='#D4A373',
                size=20,
                line=dict(width=2, color='#3B2F2F')
            )
        )
        
        fig_net = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
             )
        )
        st.plotly_chart(fig_net, use_container_width=True)
        
        st.subheader("🔥 Heatmap Association Rules (Lift)")
        pivot = rules.pivot(index='antecedents_str', columns='consequents_str', values='lift')
        fig_heat = px.imshow(pivot, color_continuous_scale=['#F5E6D3', '#D4A373', '#6F4E37', '#3B2F2F'], aspect="auto")
        fig_heat.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)


# --- TAB 3: BUNDLING ---
with tab3:
    st.subheader("💡 Rekomendasi Bundling Menu (Actionable Insights)")
    
    if rules.empty:
        st.info("Tidak ada aturan asosiasi yang cukup kuat untuk membuat rekomendasi. Coba sesuaikan parameter.")
    else:
        # Sort rules by Lift to get the best bundles
        best_rules = rules.sort_values('lift', ascending=False).head(5)
        
        st.markdown("### 🏆 Top Rekomendasi Bundling")
        
        cols = st.columns(len(best_rules) if len(best_rules) < 4 else 3)
        col_idx = 0
        
        for idx, row in best_rules.iterrows():
            with cols[col_idx % 3]:
                st.markdown(f"""
                <div style="background-color: white; padding: 15px; border-radius: 10px; border-left: 5px solid var(--accent-brown); margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="color: var(--espresso); margin-top: 0;">Paket: {row['antecedents_str']} + {row['consequents_str']}</h4>
                    <p style="margin: 5px 0;"><strong>Support:</strong> {row['support']:.2%} (sering dibeli bersama)</p>
                    <p style="margin: 5px 0;"><strong>Confidence:</strong> {row['confidence']:.2%} (peluang beli {row['consequents_str']} jika beli {row['antecedents_str']})</p>
                    <p style="margin: 5px 0;"><strong>Lift:</strong> {row['lift']:.2f} (kekuatan hubungan)</p>
                </div>
                """, unsafe_allow_html=True)
            col_idx += 1
            if col_idx == 3:
                break # Just show top 3 as cards
                
        st.markdown("---")
        st.markdown("### 🧮 Simulator Bundling")
        
        all_items_in_rules = sorted(list(set(rules['antecedents_str'].unique().tolist() + rules['consequents_str'].unique().tolist())))
        
        selected_item = st.selectbox("Pilih Menu Utama:", all_items_in_rules)
        
        recommendations = rules[rules['antecedents_str'] == selected_item].sort_values('lift', ascending=False)
        
        if recommendations.empty:
            st.info(f"Belum ada rekomendasi pendamping yang kuat untuk {selected_item} berdasarkan parameter saat ini.")
        else:
            top_rec = recommendations.iloc[0]
            st.success(f"**Rekomendasi Terbaik:** Tawarkan **{top_rec['consequents_str']}** kepada pelanggan yang membeli **{selected_item}**.")
            st.write("Statistik Hubungan:")
            st.write(f"- Probabilitas mereka akan membeli {top_rec['consequents_str']}: **{top_rec['confidence']:.1%}**")
            st.write(f"- Peningkatan penjualan {top_rec['consequents_str']} karena promosi ini: **{top_rec['lift']:.2f}x lipat** dari penjualan biasa.")
            
            st.write("Ide Promosi:")
            st.info(f"Dapatkan diskon 15% untuk **{top_rec['consequents_str']}** setiap pembelian **{selected_item}**!")

# --- TAB 4: DOWNLOAD ---
with tab4:
    st.subheader("📥 Unduh Laporan & Dataset")
    
    if not rules.empty:
        st.markdown("Download hasil rekomendasi bundling bisnis berdasarkan association rules terbaik.")
        
        # Prepare business report dataframe
        report_df = rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].copy()
        report_df.columns = ['Menu Utama', 'Menu Pendamping', 'Support', 'Confidence', 'Lift Ratio']
        report_df['Rekomendasi Aksi'] = "Buat paket bundling " + report_df['Menu Utama'] + " dan " + report_df['Menu Pendamping']
        report_df = report_df.sort_values('Lift Ratio', ascending=False)
        
        st.dataframe(report_df, use_container_width=True)
        
        csv = report_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Download Hasil Rekomendasi (CSV)",
            data=csv,
            file_name='rekomendasi_bundling_coffee_shop.csv',
            mime='text/csv',
        )
    else:
        st.warning("Belum ada rules yang dihasilkan untuk diunduh.")

    st.markdown("---")
    st.markdown("### Ringkasan Strategi Bisnis")
    if not rules.empty:
        best = rules.loc[rules['lift'].idxmax()]
        st.markdown(f"""
        Berdasarkan analisis algoritma Apriori, strategi terbaik yang dapat diimplementasikan segera adalah:
        - **Fokus Bundling Utama:** Menggabungkan **{best['antecedents_str']}** dan **{best['consequents_str']}**.
        - **Alasan:** Kombinasi ini memiliki Lift Ratio tertinggi ({best['lift']:.2f}), yang berarti pelanggan sangat terdorong untuk membeli {best['consequents_str']} jika mereka sudah membeli {best['antecedents_str']}.
        - **Saran Penempatan Menu:** Tempatkan {best['consequents_str']} di dekat kasir atau tawarkan secara verbal saat pelanggan memesan {best['antecedents_str']}.
        """)
    else:
        st.write("Silakan atur parameter untuk mendapatkan ringkasan.")
