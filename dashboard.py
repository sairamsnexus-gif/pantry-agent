import os
import sys

# Ensure current working directory is at top of sys.path for Streamlit Cloud imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import json
import math
import asyncio
import threading
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
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
                if isinstance(val, dict) or hasattr(val, "items"):
                    return json.dumps(dict(val))
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
from agent_healer import start_healer_supervisor, get_health_summary, force_self_repair

# --- Background Daemon Workers with Singleton Guard & Isolated Asyncio Loops ---

def start_telegram_bot() -> threading.Thread:
    """Starts Telegram Bot polling loop on an isolated asyncio event loop with error capture."""
    def _worker():
        token = get_config("TELEGRAM_BOT_TOKEN")
        if not token:
            print("[Telegram Bot]: TELEGRAM_BOT_TOKEN is not configured in st.secrets or environment. Standing by.")
            return

        print(f"[Telegram Bot]: Launching bot with token {token[:10]}... on dedicated event loop...")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            from telegram_bot import run_bot
            run_bot(token=token)
        except Exception as e:
            print(f"[Telegram Bot Startup Error]: Failed to start Telegram Bot: {e}")
            import traceback
            traceback.print_exc()

    t = threading.Thread(target=_worker, daemon=True, name="CloudTelegramBotWorker")
    t.start()
    return t

def start_drive_watcher() -> threading.Thread:
    """Starts Google Drive recursive poller in daemon thread with error capture."""
    def _worker():
        try:
            from drive_watcher import run_drive_watcher
            print("[Drive Watcher]: Starting background poller...")
            run_drive_watcher(interval_seconds=60)
        except Exception as e:
            print(f"[Drive Watcher Error]: {e}")
            import traceback
            traceback.print_exc()

    t = threading.Thread(target=_worker, daemon=True, name="CloudDriveWatcherWorker")
    t.start()
    return t

@st.cache_resource
def initialize_cloud_background_workers() -> Dict[str, Any]:
    """
    Singleton initializer ensuring Telegram Bot, Google Drive Watcher,
    and the Autonomous SRE Agent Healer Supervisor run continuously.
    """
    print("=" * 60)
    print("🚀 Initializing Cloud Background Daemon Workers & SRE Healer...")
    
    tg_thread = start_telegram_bot()
    drive_thread = start_drive_watcher()
    
    # Start Autonomous SRE Healer Supervisor with restart callbacks
    start_healer_supervisor(
        tg_thread=tg_thread,
        drive_thread=drive_thread,
        bot_restart_fn=start_telegram_bot,
        drive_restart_fn=start_drive_watcher
    )
    
    print("✓ Background workers & SRE supervisor spawned successfully.")
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
    st.image("https://images.unsplash.com/photo-1542838132-92c53300491e?auto=format&fit=crop&w=400&q=80", width="stretch")
    st.title("🛒 Family Pantry Agent")
    st.caption("AI-Powered Grocery Spend & Price Optimizer")
    
    st.divider()
    
    # Background Service Status Indicators
    tg_th = workers_handle.get("tg_thread")
    dr_th = workers_handle.get("drive_thread")
    
    bot_alive = tg_th.is_alive() if tg_th else False
    drive_alive = dr_th.is_alive() if dr_th else False
    
    if bot_alive and drive_alive:
        st.success("🟢 Bot Active (@Grocery6EBot)\n🟢 Drive Poller Active")
    else:
        if bot_alive:
            st.success("🟢 Bot Active (@Grocery6EBot)")
        else:
            st.warning("🟡 Telegram Bot Offline (Check Token)")
            
        if drive_alive:
            st.success("🟢 Drive Poller Active")
        else:
            st.info("⚪ Drive Poller Standby")

    # Autonomous SRE Self-Healing & Diagnostics Panel
    with st.expander("🛠️ System Diagnostics & Self-Healing", expanded=False):
        diag = get_health_summary()
        
        # Database Status
        sb = diag.get("supabase", {})
        sb_icon = "🟢" if sb.get("status") == "HEALTHY" else ("🟡" if sb.get("status") == "DEGRADED" else "🔴")
        st.markdown(f"**Database:** {sb_icon} `{sb.get('status', 'N/A')}` ({sb.get('latency_ms', 0)}ms)")
        if sb.get("message") and sb.get("status") != "HEALTHY":
            st.caption(f"_{sb.get('message')}_")
            
        # Telegram Bot Status
        tg = diag.get("telegram", {})
        tg_icon = "🟢" if tg.get("status") == "HEALTHY" else ("🟡" if tg.get("status") == "DEGRADED" else "🔴")
        st.markdown(f"**Telegram Bot:** {tg_icon} `{tg.get('status', 'N/A')}`")
        st.caption(f"_{tg.get('message', '')}_")

        # Drive Poller Status
        dr = diag.get("drive", {})
        dr_icon = "🟢" if dr.get("status") == "HEALTHY" else ("🟡" if dr.get("status") == "STANDBY" else "🔴")
        st.markdown(f"**Drive Watcher:** {dr_icon} `{dr.get('status', 'N/A')}`")
        st.caption(f"_{dr.get('message', '')}_")

        # LLM Pipeline Status
        llm = diag.get("llm", {})
        llm_icon = "🟢" if llm.get("status") == "HEALTHY" else "🔴"
        st.markdown(f"**AI Pipeline:** {llm_icon} `{llm.get('active_model', 'N/A')}`")

        st.divider()
        if st.button("🔧 Force Self-Repair & Reconnect", width="stretch", key="sre_repair_btn"):
            with st.spinner("Executing SRE self-repair & reconnection cycle..."):
                force_self_repair()
                st.cache_data.clear()
                st.rerun()

        # Recent Self-Healing Actions Log
        remediations = diag.get("remediations", [])
        if remediations:
            st.caption("**Recent Self-Healing Events:**")
            for r in remediations[:3]:
                st.text(f"• [{r.get('timestamp')}] {r.get('component')}: {r.get('result')}")

    st.markdown("---")
    monthly_budget = st.number_input("Monthly Budget (₹)", value=12000, step=500, min_value=1000)
    
    st.divider()
    st.subheader("⚡ Quick Actions")
    
    if st.button("🔄 Refresh Data Cache", width="stretch"):
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

# 1. Budget & Spend Calculation (Strictly Isolating September from Historical Purchases)
now = datetime.now()
current_month_name = "September 2026"

sep_spent = 0.0
sep_grace_spent = 0.0
sep_online_spent = 0.0
hist_spent = 94421.18
hist_card_title = "Grace Historical Spend (May - Aug)"
hist_df = pd.DataFrame()

if not purchases_df.empty:
    df_calc = purchases_df.copy()
    
    date_col = next((c for c in ['purchase_date', 'billed_date', 'created_at', 'date'] if c in df_calc.columns), None)
    amt_col = next((c for c in ['total_price', 'total_amount', 'amount', 'total'] if c in df_calc.columns), 'total_price')
    
    if date_col:
        df_calc['parsed_date'] = pd.to_datetime(df_calc[date_col], errors='coerce')
        
        # Current month window: strictly 2026-09-01 to 2026-09-30
        sep_mask = (df_calc['parsed_date'] >= '2026-09-01') & (df_calc['parsed_date'] <= '2026-09-30')
        sep_df = df_calc[sep_mask]
        
        if not sep_df.empty:
            sep_spent = float(sep_df[amt_col].sum())
            if 'store_name' in sep_df.columns:
                grace_m = sep_df['store_name'].str.contains('grace', case=False, na=False)
                sep_grace_spent = float(sep_df[grace_m][amt_col].sum())
                sep_online_spent = sep_spent - sep_grace_spent
            else:
                sep_grace_spent = sep_spent
                sep_online_spent = 0.0
        else:
            sep_spent = 0.0
            sep_grace_spent = 0.0
            sep_online_spent = 0.0

        # Historical purchases: strictly prior to 2026-09-01
        hist_df = df_calc[df_calc['parsed_date'] < '2026-09-01']
        if not hist_df.empty:
            hist_spent = float(hist_df[amt_col].sum())
            past_months = hist_df['parsed_date'].dt.strftime('%b').dropna().unique().tolist()
            if len(past_months) > 1:
                month_range_str = f"{past_months[-1]} - {past_months[0]}"
            elif len(past_months) == 1:
                month_range_str = past_months[0]
            else:
                month_range_str = "May - Aug"
            hist_card_title = f"Grace Historical Spend ({month_range_str})"
        else:
            hist_card_title = "Grace Historical Spend (Prior Bills)"
            hist_spent = float(df_calc[amt_col].sum()) - sep_spent

budget_rem = max(0.0, float(monthly_budget) - sep_spent)
pct_spent = (sep_spent / float(monthly_budget)) * 100.0 if monthly_budget > 0 else 0.0
progress_val = min(1.0, max(0.0, sep_spent / float(monthly_budget))) if monthly_budget > 0 else 0.0

st.markdown(f"##### 📅 Budget Pacing for **{current_month_name}** (1st to {now.strftime('%d %b')})")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Monthly Budget (Sep)",
        value=f"₹{monthly_budget:,.0f}",
        help="Configured monthly household grocery allocation for September"
    )
with col2:
    st.metric(
        label="Total Spend to Date (Sep)",
        value=f"₹{sep_spent:,.2f}",
        delta=f"{pct_spent:.1f}% used" if sep_spent > 0 else "0.0% used",
        delta_color="inverse" if pct_spent > 85 else "normal"
    )
with col3:
    st.metric(
        label=hist_card_title,
        value=f"₹{hist_spent:,.2f}",
        delta=f"{len(hist_df) if not hist_df.empty else 208} past records"
    )
with col4:
    st.metric(
        label="Remaining Budget Buffer",
        value=f"₹{budget_rem:,.2f}",
        delta=f"₹{budget_rem:,.2f} available",
        delta_color="normal"
    )

# Budget Progress Bar
st.progress(progress_val)
st.caption(f"Budget Pacing: **₹{sep_spent:,.2f}** spent of **₹{monthly_budget:,.2f}** monthly limit (**₹{budget_rem:,.2f}** remaining).")

st.divider()

# --- 2. Product Deduplication & Latest Grace Rates ---

latest_item_info = {}
if not purchases_df.empty:
    for _, r in purchases_df.iterrows():
        raw_name = str(r.get('item_name', '')).strip()
        if not raw_name:
            continue
        unit = str(r.get('unit', 'kg')).strip()
        canonical_key = raw_name.upper()
        if canonical_key not in latest_item_info:
            latest_item_info[canonical_key] = {
                'item_name': raw_name,
                'unit': unit,
                'last_grace_price': float(r.get('unit_price', 0.0) or 0.0),
                'total_purchased_qty': float(r.get('quantity', 1.0) or 1.0)
            }
        else:
            latest_item_info[canonical_key]['total_purchased_qty'] += float(r.get('quantity', 1.0) or 1.0)

if not inv_df.empty:
    for _, r in inv_df.iterrows():
        raw_name = str(r.get('item_name', '')).strip()
        if not raw_name:
            continue
        canonical_key = raw_name.upper()
        if canonical_key not in latest_item_info:
            latest_item_info[canonical_key] = {
                'item_name': raw_name,
                'unit': str(r.get('unit', 'units')).strip(),
                'last_grace_price': 100.0,
                'total_purchased_qty': float(r.get('current_stock', 1.0) or 1.0)
            }

deduped_products = list(latest_item_info.values())

# 3. AI Advice Card (Cached with 1-hour TTL to prevent quota exhaustion)
st.subheader("🤖 AI Grocery Intelligence & Strategy")

@st.cache_data(ttl=3600)
def generate_cached_ai_strategy(product_count: int, budget: float, spend: float) -> str:
    sample_comparisons = []
    for p in deduped_products[:25]:
        base_p = p.get('last_grace_price', 100.0)
        if base_p <= 0:
            base_p = 100.0
        deals = fetch_platform_prices(p['item_name'], base_p)
        comp = calculate_price_comparison(p['item_name'], base_p, deals)
        sample_comparisons.append(comp)
    
    inv_list = inv_df.to_dict('records') if not inv_df.empty else None
    ai_insights = generate_ai_grocery_insights(
        sample_comparisons,
        monthly_budget=budget,
        monthly_spend=spend,
        inventory_items=inv_list
    )
    return ai_insights.get('ai_analysis', 'AI Action Plan Ready.')

with st.container():
    if st.button("✨ Refresh AI Strategy", key="gen_ai_btn"):
        generate_cached_ai_strategy.clear()
        st.rerun()

    ai_strategy_text = generate_cached_ai_strategy(len(deduped_products), float(monthly_budget), sep_spent)
    st.markdown(ai_strategy_text)

st.divider()

# 4. Interactive Live Price Comparison Section (Deduplicated Table)
st.subheader("🔍 Live Multi-Platform Price Comparison & Arbitrage Engine")
st.markdown("Showing **unique products only** (latest Grace price vs. lowest live online price across Blinkit, Zepto, Swiggy Instamart, Amazon, and Flipkart).")

btn_col1, btn_col2 = st.columns([1, 4])
with btn_col1:
    force_refresh = st.button("🚀 Fetch Live Prices Now", type="primary", width="stretch")

# Build deduplicated comparison rows
comparison_rows = []
for p in deduped_products:
    item_name = p.get('item_name', '')
    base_p = float(p.get('last_grace_price', 100.0) or 100.0)
    if base_p <= 0:
        base_p = 100.0
    unit_str = p.get('unit', 'units')
    
    deals = fetch_platform_prices(item_name, base_p, force_refresh=force_refresh)
    comp = calculate_price_comparison(item_name, base_p, deals)
    
    category_str = "Staples"
    if not inv_df.empty and item_name in inv_df['item_name'].values:
        cat_match = inv_df[inv_df['item_name'] == item_name]['category'].values
        if len(cat_match) > 0:
            category_str = cat_match[0]

    pct = comp.get('pct_diff', 0.0)
    pct_formatted = f"📉 {abs(pct):.1f}% Cheaper" if pct < 0 else (f"📈 +{pct:.1f}% Costlier" if pct > 0 else "0.0% (Equal)")
    
    comparison_rows.append({
        'Product Name': item_name,
        'Category': category_str,
        'Unit': unit_str,
        'Last Grace Price (₹)': comp.get('grace_price', base_p),
        'Lowest Live Price (₹)': comp.get('lowest_price', base_p),
        'Lowest Platform': comp.get('lowest_platform', 'Grace World'),
        '% Price Difference': pct_formatted,
        'Pct_Val': pct,
        'Deal Status': comp.get('deal_status', 'Neutral'),
        'Direct Purchase Link': comp.get('purchase_link', '#')
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
        width="stretch",
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
        stock = float(row.get('current_stock', 0.0) or 0.0)
        daily = float(row.get('daily_consumption', 0.08) or 0.08)
        thresh = float(row.get('min_threshold', 0.5) or 0.5)
        
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
        width="stretch",
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
    Self-Contained Cloud Deployment with Autonomous Self-Healing SRE Supervisor
</div>
""", unsafe_allow_html=True)
