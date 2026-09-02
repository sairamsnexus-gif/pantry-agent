import os
import sys
import math
import asyncio
import threading
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('.env')
load_dotenv('.env.env')

# Helper to support both Streamlit Community Cloud st.secrets and local .env
def get_config(key: str, default: str = "") -> str:
    # 1. Check st.secrets first for cloud deployments
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is not None:
                return str(val).strip()
    except Exception:
        pass
    
    # 2. Fallback to os.environ / .env
    val = os.environ.get(key)
    if val:
        return val.strip()
        
    return default

# Sync all configuration keys into environment so imported modules can access them
CONFIG_KEYS = [
    "GEMINI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "DRIVE_FOLDER_ID",
    "SERVICE_ACCOUNT_JSON",
    "GCP_SERVICE_ACCOUNT"
]

for k in CONFIG_KEYS:
    val = get_config(k)
    if val:
        os.environ[k] = val

SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_KEY = get_config("SUPABASE_KEY")
GEMINI_API_KEY = get_config("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = get_config("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_config("TELEGRAM_CHAT_ID")
DRIVE_FOLDER_ID = get_config("DRIVE_FOLDER_ID", "1CofRa3fSzj8OEE28OvZHefrffRuMM6cN")

# Import system modules
from price_fetcher import fetch_platform_prices, get_best_online_deal, normalize_product_name
from insights import calculate_price_comparison, generate_ai_grocery_insights

# --- Background Daemon Workers with Singleton Guard ---

def _run_telegram_bot_loop():
    """Isolated thread worker running Telegram Bot & APScheduler on its own event loop."""
    if not TELEGRAM_BOT_TOKEN:
        print("[Telegram Bot Worker]: TELEGRAM_BOT_TOKEN not provided. Skipping bot.")
        return

    # Dedicated asyncio loop for this background thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from telegram_bot import build_telegram_app, setup_scheduler
        app = build_telegram_app()
        scheduler = setup_scheduler(app)
        print("[Telegram Bot Worker]: Starting polling loop on daemon thread...")
        app.run_polling(drop_pending_updates=True, close_loop=False)
    except Exception as e:
        print(f"[Telegram Bot Worker Error]: {e}")

def _run_drive_watcher_loop():
    """Isolated thread worker running recursive Google Drive polling."""
    try:
        from drive_watcher import run_drive_watcher
        print("[Drive Watcher Worker]: Starting recursive poller on daemon thread...")
        run_drive_watcher(interval_seconds=60)
    except Exception as e:
        print(f"[Drive Watcher Worker Error]: {e}")

@st.cache_resource
def initialize_cloud_background_workers() -> Dict[str, Any]:
    """
    Singleton initializer ensuring Telegram Bot and Google Drive Watcher
    run exactly ONCE in background daemon threads across all user reruns/sessions.
    """
    print("=" * 60)
    print("🚀 Initializing Cloud Background Daemon Workers (@st.cache_resource)...")
    
    tg_thread = threading.Thread(
        target=_run_telegram_bot_loop,
        daemon=True,
        name="CloudTelegramBotWorker"
    )
    tg_thread.start()
    
    drive_thread = threading.Thread(
        target=_run_drive_watcher_loop,
        daemon=True,
        name="CloudDriveWatcherWorker"
    )
    drive_thread.start()
    
    print("✓ Background workers launched in non-blocking daemon threads.")
    print("=" * 60)
    
    return {
        "tg_thread": tg_thread,
        "drive_thread": drive_thread,
        "initialized_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Start background workers (runs only once per server lifetime)
workers_handle = initialize_cloud_background_workers()

# --- Streamlit Page Configuration ---

st.set_page_config(
    page_title="Family Grocery Intelligence System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)

client = get_supabase_client()

# --- Data Fetching ---

@st.cache_data(ttl=60)
def load_inventory_data():
    if not client:
        return pd.DataFrame()
    try:
        res = client.table("inventory").select("*").order("item_name").execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        st.error(f"Error loading inventory: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_purchase_history():
    if not client:
        return pd.DataFrame()
    try:
        res = client.table("purchase_history").select("*").order("id", desc=True).execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        st.error(f"Error loading purchases: {e}")
        return pd.DataFrame()

inv_df = load_inventory_data()
purchases_df = load_purchase_history()

# --- Sidebar Controls & Live Service Status ---

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1542838132-92c53300491e?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.title("🛒 Family Pantry Agent")
    st.caption("AI-Powered Grocery Spend & Price Optimizer")
    
    st.divider()
    
    # Background Service Status Indicator
    bot_alive = workers_handle["tg_thread"].is_alive()
    drive_alive = workers_handle["drive_thread"].is_alive()
    
    if bot_alive and drive_alive:
        st.success("🟢 Cloud Bot & Drive Poller Active")
    elif bot_alive or drive_alive:
        st.warning(f"🟡 Services: {'Bot Active' if bot_alive else 'Bot Offline'} | {'Drive Poller Active' if drive_alive else 'Drive Offline'}")
    else:
        st.error("🔴 Background Workers Offline")

    st.markdown("---")
    monthly_budget = st.number_input("Monthly Budget (₹)", value=12000, step=500, min_value=1000)
    
    st.divider()
    st.subheader("⚡ Quick Actions")
    
    if st.button("🔄 Refresh Data Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📱 Cloud Integrations")
    st.markdown("✅ **Telegram Bot:** `@Grocery6EBot`")
    st.markdown("✅ **Drive Auto-Ingestion:** Folder `1Cof...`")
    st.markdown("✅ **Friday 9AM IST Checklist:** Scheduled")
    st.markdown("✅ **Supabase Database:** Connected")

# --- Main Dashboard ---

st.markdown('<div class="main-header">🏡 Family Grocery Intelligence & Pantry Command</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time price arbitrage across Grace Supermarket, Blinkit, Zepto, Swiggy Instamart, Amazon & Flipkart</div>', unsafe_allow_html=True)

# 1. Current Month Budget Pacing (Strictly Current Calendar Month)
now = datetime.now()
current_year = now.year
current_month = now.month
current_month_name = now.strftime('%B %Y')

current_month_spent = 0.0
current_grace_spent = 0.0
current_online_spent = 0.0

if not purchases_df.empty:
    df_temp = purchases_df.copy()
    if 'created_at' in df_temp.columns:
        df_temp['created_date'] = pd.to_datetime(df_temp['created_at'], errors='coerce')
        month_mask = (df_temp['created_date'].dt.year == current_year) & (df_temp['created_date'].dt.month == current_month)
        current_month_df = df_temp[month_mask]
    else:
        current_month_df = df_temp

    if not current_month_df.empty:
        current_month_spent = float(current_month_df['total_price'].sum())
        grace_mask = current_month_df['store_name'].str.contains('grace', case=False, na=False)
        current_grace_spent = float(current_month_df[grace_mask]['total_price'].sum())
        current_online_spent = current_month_spent - current_grace_spent
    else:
        current_month_spent = 0.0
        current_grace_spent = 0.0
        current_online_spent = 0.0
else:
    current_month_spent = 0.0
    current_grace_spent = 0.0
    current_online_spent = 0.0

budget_rem = max(0.0, float(monthly_budget) - current_month_spent)
pct_spent = min(100.0, (current_month_spent / float(monthly_budget)) * 100.0) if monthly_budget > 0 else 0.0

st.markdown(f"##### 📅 Budget Pacing for **{current_month_name}** (1st to {now.strftime('%d %b')})")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label=f"Monthly Budget ({now.strftime('%b')})",
        value=f"₹{monthly_budget:,.0f}",
        help="Configured monthly household grocery envelope"
    )
with col2:
    st.metric(
        label=f"Total Spend to Date ({now.strftime('%b')})",
        value=f"₹{current_month_spent:,.2f}",
        delta=f"{pct_spent:.1f}% of budget used",
        delta_color="inverse" if pct_spent > 85 else "normal"
    )
with col3:
    st.metric(
        label="Grace Offline Spend",
        value=f"₹{current_grace_spent:,.2f}",
        delta=f"{(current_grace_spent/current_month_spent*100) if current_month_spent>0 else 0:.0f}% offline share"
    )
with col4:
    st.metric(
        label="Remaining Budget Buffer",
        value=f"₹{budget_rem:,.2f}",
        delta=f"₹{budget_rem:,.0f} available"
    )

# Budget Progress Bar
st.progress(pct_spent / 100.0)
st.caption(f"Budget Health: **₹{current_month_spent:,.2f}** spent in {current_month_name} of **₹{monthly_budget:,.2f}** limit (**₹{budget_rem:,.2f}** remaining buffer).")

st.divider()

# --- 2. Product Deduplication & Latest Grace Rates ---

latest_item_info = {}
if not purchases_df.empty:
    for _, r in purchases_df.iterrows():
        raw_name = str(r['item_name']).strip()
        unit = str(r.get('unit', 'kg')).strip()
        canonical_key = raw_name.upper()
        if canonical_key not in latest_item_info:
            latest_item_info[canonical_key] = {
                'item_name': raw_name,
                'unit': unit,
                'last_grace_price': float(r.get('unit_price', 0.0) or 0.0),
                'total_purchased_qty': float(r.get('quantity', 1.0))
            }
        else:
            latest_item_info[canonical_key]['total_purchased_qty'] += float(r.get('quantity', 1.0))

if not inv_df.empty:
    for _, r in inv_df.iterrows():
        raw_name = str(r['item_name']).strip()
        canonical_key = raw_name.upper()
        if canonical_key not in latest_item_info:
            latest_item_info[canonical_key] = {
                'item_name': raw_name,
                'unit': str(r.get('unit', 'units')).strip(),
                'last_grace_price': 100.0,
                'total_purchased_qty': float(r.get('current_stock', 1.0))
            }

deduped_products = list(latest_item_info.values())

# 3. AI Advice Card
st.subheader("🤖 AI Grocery Intelligence & Strategy")

with st.container():
    if st.button("✨ Refresh AI Strategy", key="gen_ai_btn"):
        st.session_state['ai_strategy'] = None

    if 'ai_strategy' not in st.session_state or st.session_state['ai_strategy'] is None:
        with st.spinner("Synthesizing concise grocery advice with Gemini Flash..."):
            sample_comparisons = []
            for p in deduped_products[:25]:
                base_p = p['last_grace_price'] if p['last_grace_price'] > 0 else 100.0
                deals = fetch_platform_prices(p['item_name'], base_p)
                comp = calculate_price_comparison(p['item_name'], base_p, deals)
                sample_comparisons.append(comp)
            
            ai_insights = generate_ai_grocery_insights(
                sample_comparisons,
                monthly_budget=float(monthly_budget),
                monthly_spend=current_month_spent
            )
            st.session_state['ai_strategy'] = ai_insights['ai_analysis']

    st.info(st.session_state['ai_strategy'])

st.divider()

# 4. Interactive Live Price Comparison Section (Deduplicated Table)
st.subheader("🔍 Live Multi-Platform Price Comparison & Arbitrage Engine")
st.markdown("Showing **unique products only** (latest Grace price vs. lowest live online price across Blinkit, Zepto, Swiggy Instamart, Amazon, and Flipkart).")

btn_col1, btn_col2 = st.columns([1, 4])
with btn_col1:
    force_refresh = st.button("🚀 Fetch Live Prices Now", type="primary", use_container_width=True)

# Build deduplicated comparison rows
comparison_rows = []
for p in deduped_products:
    item_name = p['item_name']
    base_p = p['last_grace_price'] if p['last_grace_price'] > 0 else 100.0
    unit_str = p['unit']
    
    deals = fetch_platform_prices(item_name, base_p, force_refresh=force_refresh)
    comp = calculate_price_comparison(item_name, base_p, deals)
    
    category_str = "Staples"
    if not inv_df.empty and item_name in inv_df['item_name'].values:
        cat_match = inv_df[inv_df['item_name'] == item_name]['category'].values
        if len(cat_match) > 0:
            category_str = cat_match[0]

    pct = comp['pct_diff']
    pct_formatted = f"📉 {abs(pct):.1f}% Cheaper" if pct < 0 else (f"📈 +{pct:.1f}% Costlier" if pct > 0 else "0.0% (Equal)")
    
    comparison_rows.append({
        'Product Name': item_name,
        'Category': category_str,
        'Unit': unit_str,
        'Last Grace Price (₹)': comp['grace_price'],
        'Lowest Live Price (₹)': comp['lowest_price'],
        'Lowest Platform': comp['lowest_platform'],
        '% Price Difference': pct_formatted,
        'Pct_Val': pct,
        'Deal Status': comp['deal_status'],
        'Direct Purchase Link': comp['purchase_link']
    })

comp_df = pd.DataFrame(comparison_rows)

if not comp_df.empty:
    comp_df = comp_df.drop_duplicates(subset=['Product Name'], keep='first')

    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    with f_col1:
        search_query = st.text_input("🔎 Search Product Name", placeholder="e.g. Toor Dal, Sunflower Oil, Salt, Atta...")
    with f_col2:
        cat_filter = st.selectbox("Filter Category", ["All"] + sorted(list(comp_df['Category'].unique())))
    with f_col3:
        status_filter = st.selectbox("Deal Filter", ["All Deals", "🟢 Best Buys Online", "🔴 Do Not Buy Online", "⚪ Fair Deals"])

    filtered_df = comp_df.copy()
    if search_query:
        filtered_df = filtered_df[filtered_df['Product Name'].str.contains(search_query, case=False, na=False)]
    if cat_filter != "All":
        filtered_df = filtered_df[filtered_df['Category'] == cat_filter]
    if status_filter == "🟢 Best Buys Online":
        filtered_df = filtered_df[filtered_df['Deal Status'] == 'Best Buy Online']
    elif status_filter == "🔴 Do Not Buy Online":
        filtered_df = filtered_df[filtered_df['Deal Status'] == 'Do Not Buy Online']
    elif status_filter == "⚪ Fair Deals":
        filtered_df = filtered_df[filtered_df['Deal Status'] == 'Fair Deal']

    display_df = filtered_df[['Product Name', 'Category', 'Unit', 'Last Grace Price (₹)', 'Lowest Live Price (₹)', 'Lowest Platform', '% Price Difference', 'Deal Status', 'Direct Purchase Link']].copy()

    st.dataframe(
        display_df,
        column_config={
            "Last Grace Price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "Lowest Live Price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "Direct Purchase Link": st.column_config.LinkColumn("Purchase Online", display_text="Open Store ↗"),
            "Deal Status": st.column_config.TextColumn("Verdict")
        },
        use_container_width=True,
        hide_index=True,
        height=450
    )

    st.caption(f"Showing **{len(display_df)} unique products**. 🟢 **Best Buy Online:** Online price is ≥ 5% cheaper than Grace rates | 🔴 **Do Not Buy Online:** Online rate carries markups over Grace shelf price.")

st.divider()

# 5. Live Pantry Inventory & Runout Countdown Table
st.subheader("📦 Live Pantry Inventory & Countdown Tracker")

if not inv_df.empty:
    inv_display = inv_df.copy()
    
    today = datetime.now()
    days_list = []
    runout_dates = []
    status_icons = []
    
    for _, row in inv_display.iterrows():
        stock = float(row.get('current_stock', 0.0))
        daily = float(row.get('daily_consumption', 0.08))
        thresh = float(row.get('min_threshold', 0.5))
        
        days = int(stock / daily) if daily > 0 else 999
        days_list.append(days)
        
        target_date = today + timedelta(days=days)
        runout_dates.append(target_date.strftime('%Y-%m-%d'))
        
        if stock <= 0:
            status_icons.append("🔴 Empty")
        elif stock <= thresh or days <= 5:
            status_icons.append("🟡 Low Stock (<5d)")
        else:
            status_icons.append("🟢 Healthy")

    inv_display['Days Remaining'] = days_list
    inv_display['Estimated Runout Date'] = runout_dates
    inv_display['Stock Status'] = status_icons
    
    cols_to_show = ['item_name', 'category', 'current_stock', 'unit', 'daily_consumption', 'min_threshold', 'Days Remaining', 'Estimated Runout Date', 'Stock Status']
    inv_clean = inv_display[cols_to_show].rename(columns={
        'item_name': 'Item Name',
        'category': 'Category',
        'current_stock': 'Current Stock',
        'unit': 'Unit',
        'daily_consumption': 'Daily Burn Rate',
        'min_threshold': 'Min Threshold'
    })
    
    inv_clean = inv_clean.drop_duplicates(subset=['Item Name'], keep='first')
    inv_clean = inv_clean.sort_values(by='Days Remaining')
    
    st.dataframe(
        inv_clean,
        column_config={
            "Current Stock": st.column_config.NumberColumn(format="%.2f"),
            "Daily Burn Rate": st.column_config.NumberColumn(format="%.2f / day"),
            "Days Remaining": st.column_config.NumberColumn(format="%d days"),
            "Stock Status": st.column_config.TextColumn("Health")
        },
        use_container_width=True,
        hide_index=True,
        height=400
    )
else:
    st.info("Inventory table is currently empty. Run `seed_history.py` to seed historical stock.")

st.divider()

# Footer
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 1.5rem 0;">
    Family Grocery Intelligence System • Built with Streamlit, Supabase, Google Gemini 2.5 Flash & Python-Telegram-Bot<br>
    Self-Contained Cloud Deployment with Non-Blocking Daemon Workers
</div>
""", unsafe_allow_html=True)
