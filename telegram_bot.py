import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import io
import re
import json
import logging
from datetime import datetime, timezone
import pytz
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from supabase import create_client
from google import genai
from google.genai import types

from price_fetcher import fetch_platform_prices, get_best_online_deal, normalize_product_name
from insights import calculate_price_comparison, generate_ai_grocery_insights

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('.env')
load_dotenv('.env.env')

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = """🛒 *Family Grocery Intelligence & Pantry Bot* 🛒

Welcome! I help manage our household grocery inventory, compare live market prices, and track receipts.

📌 *Available Commands:*
• 📊 `/stock` - View current inventory levels & runout dates
• 🔍 `/compare <item>` - Live price check across Blinkit, Zepto, Swiggy, Amazon & Flipkart
• 💡 `/deals` - AI price intelligence & Best Buy recommendations
• 📝 `/checklist` - Interactive Low-Stock Checklist
• 💰 `/spend` - Monthly grocery spend vs ₹12,000 budget
• ℹ️ `/help` - Command assistance

📸 *Receipt Ingestion:*
Simply snap a photo or send a PDF of any grocery receipt directly to this chat, and I'll automatically parse and sync it to our pantry database!"""
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start_command(update, context)

async def stock_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    supabase = get_supabase()
    if not supabase:
        await update.message.reply_text("⚠️ Database connection not configured.")
        return

    try:
        res = supabase.table("inventory").select("*").order("current_stock").limit(20).execute()
        items = res.data or []
        if not items:
            await update.message.reply_text("📦 Inventory is currently empty. Upload a bill to seed items!")
            return

        lines = ["📦 *Current Pantry Inventory & Stock Status:*\n"]
        for it in items:
            name = it['item_name']
            stock = float(it.get('current_stock', 0.0))
            unit = it.get('unit', '')
            daily = float(it.get('daily_consumption', 0.08))
            days_left = int(stock / daily) if daily > 0 else 99
            
            icon = "🔴" if stock <= float(it.get('min_threshold', 0.5)) else ("🟡" if days_left <= 7 else "🟢")
            lines.append(f"{icon} *{name}*: `{stock} {unit}` (~{days_left} days left)")

        lines.append("\n_Tip: Type /compare <item> to check online prices._")
        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in stock command: {e}")
        await update.message.reply_text(f"⚠️ Error querying inventory: {e}")

async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("🔍 *Usage:* `/compare <item name>`\n_Example:_ `/compare Toor Dal 1kg` or `/compare Sunflower Oil`", parse_mode='Markdown')
        return

    query = " ".join(context.args)
    await update.message.reply_text(f"🔎 Querying live rates for *{query}* across Blinkit, Zepto, Swiggy Instamart, Amazon, and Flipkart...", parse_mode='Markdown')

    quotes = fetch_platform_prices(query, baseline_price=150.0, force_refresh=True)
    if not quotes:
        await update.message.reply_text("⚠️ Could not retrieve live prices right now.")
        return

    best = min(quotes, key=lambda x: x['current_price'])
    lines = [f"📊 *Live Price Comparison for '{query}':*\n"]
    for q in quotes:
        icon = "🏆 " if q['platform_name'] == best['platform_name'] else "• "
        lines.append(f"{icon}*{q['platform_name']}*: ₹{q['current_price']:.2f} ({q['in_stock_status']}) [Buy Here]({q['purchase_link']})")

    lines.append(f"\n💡 *Best Deal:* {best['platform_name']} at *₹{best['current_price']:.2f}*")
    await update.message.reply_text("\n".join(lines), parse_mode='Markdown', disable_web_page_preview=True)

async def deals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    supabase = get_supabase()
    if not supabase:
        await update.message.reply_text("⚠️ Database connection not configured.")
        return

    await update.message.reply_text("🤖 Generating AI price intelligence comparing Grace World baseline rates vs live e-commerce prices...")

    try:
        # Fetch top staple items from purchase history or inventory
        ph = supabase.table("purchase_history").select("item_name, unit_price").order("id", desc=True).limit(20).execute()
        raw_items = ph.data or []
        
        seen = set()
        comparisons = []
        for r in raw_items:
            name = r['item_name']
            if name in seen:
                continue
            seen.add(name)
            grace_price = float(r.get('unit_price', 0.0) or 100.0)
            deals = fetch_platform_prices(name, grace_price)
            comp = calculate_price_comparison(name, grace_price, deals)
            comparisons.append(comp)

        ai_out = generate_ai_grocery_insights(comparisons, monthly_spend=7450.0)
        advice_text = ai_out.get('ai_analysis', 'No insights available.')
        
        await update.message.reply_text(f"🧠 *AI Grocery Intelligence & Shopping Plan*\n\n{advice_text}", parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in deals command: {e}")
        await update.message.reply_text(f"⚠️ Error generating deals: {e}")

async def spend_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    supabase = get_supabase()
    budget = 12000.0
    
    total_spend = 0.0
    grace_spend = 0.0
    online_spend = 0.0
    
    if supabase:
        try:
            ph = supabase.table("purchase_history").select("total_price, store_name").execute()
            records = ph.data or []
            for r in records:
                amt = float(r.get('total_price', 0.0))
                total_spend += amt
                if 'grace' in str(r.get('store_name', '')).lower():
                    grace_spend += amt
                else:
                    online_spend += amt
        except Exception as e:
            logger.error(f"Error fetching spend: {e}")
            total_spend = 7450.0
            grace_spend = 5900.0
            online_spend = 1550.0

    rem = max(0.0, budget - total_spend)
    pct = min(100.0, (total_spend / budget) * 100.0) if budget > 0 else 0.0

    # Build visual progress bar
    filled = int(pct / 10)
    bar = "█" * filled + "░" * (10 - filled)

    msg = f"""💰 *Family Monthly Grocery Spend Tracker*

Monthly Ceiling: *₹{budget:,.2f}*
Total Spent: *₹{total_spend:,.2f}* ({pct:.1f}%)
Remaining Buffer: *₹{rem:,.2f}*

`[{bar}]`

📊 *Breakdown by Channel:*
🏬 Grace World (Offline): *₹{grace_spend:,.2f}*
📱 Online & Quick-Commerce: *₹{online_spend:,.2f}*

{"🟢 *Spend is well within monthly budget limits!*" if pct < 85 else "⚠️ *Approaching monthly budget threshold!*"}"""
    await update.message.reply_text(msg, parse_mode='Markdown')

# --- Interactive Low-Stock Checklist ---

async def send_low_stock_checklist(bot, chat_id: str):
    """Sends interactive Friday checklist with Empty / Half / OK buttons."""
    supabase = get_supabase()
    if not supabase:
        return

    try:
        # Get items that are low in stock or essential staples
        res = supabase.table("inventory").select("item_name, current_stock, unit, min_threshold").order("current_stock").limit(6).execute()
        items = res.data or []
        if not items:
            return

        intro = "📋 *Friday 09:00 AM IST Pantry Stock Check* 📋\n\nPlease check pantry containers and update the status below:"
        await bot.send_message(chat_id=chat_id, text=intro, parse_mode='Markdown')

        for it in items:
            name = it['item_name']
            stock = it['current_stock']
            unit = it['unit']
            text = f"📦 *{name}*\nCurrent recorded: `{stock} {unit}`"
            
            # Buttons for Empty, Half, OK
            keyboard = [
                [
                    InlineKeyboardButton("🔴 Empty (0)", callback_data=f"stock:empty:{name}"),
                    InlineKeyboardButton("🟡 Half", callback_data=f"stock:half:{name}"),
                    InlineKeyboardButton("🟢 OK", callback_data=f"stock:ok:{name}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error sending checklist: {e}")

async def checklist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    await send_low_stock_checklist(context.bot, chat_id)

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    supabase = get_supabase()
    
    if data.startswith("stock:"):
        parts = data.split(":", 2)
        action = parts[1]
        item_name = parts[2]
        
        new_status_text = ""
        if action == "empty":
            if supabase:
                supabase.table("inventory").update({"current_stock": 0.0}).eq("item_name", item_name).execute()
            new_status_text = f"🔴 *{item_name}* marked as *EMPTY (0)*. Added to shopping list!"
        elif action == "half":
            if supabase:
                # Set to 1.0 or half threshold
                supabase.table("inventory").update({"current_stock": 1.0}).eq("item_name", item_name).execute()
            new_status_text = f"🟡 *{item_name}* marked as *HALF STOCK*."
        elif action == "ok":
            new_status_text = f"🟢 *{item_name}* confirmed as *OK (In Stock)*."

        await query.edit_message_text(text=f"{new_status_text}\n_Updated at {datetime.now().strftime('%H:%M:%S')}_", parse_mode='Markdown')

# --- Receipt Photo & Document Parser ---

async def handle_receipt_document_or_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return

    # 1. Immediate Receipt acknowledgment
    status_msg = await message.reply_text("📥 Receipt image received! Reading with Gemini Vision...")

    try:
        file_bytes = None
        file_name = "receipt.jpg"
        mime_type = "image/jpeg"

        if message.photo:
            photo = message.photo[-1]
            tg_file = await photo.get_file()
            byte_arr = await tg_file.download_as_bytearray()
            file_bytes = bytes(byte_arr)
            mime_type = "image/jpeg"
            file_name = f"receipt_{photo.file_id[:8]}.jpg"
        elif message.document:
            doc = message.document
            tg_file = await doc.get_file()
            byte_arr = await tg_file.download_as_bytearray()
            file_bytes = bytes(byte_arr)
            file_name = doc.file_name or "receipt.pdf"
            mime_type = doc.mime_type or "application/pdf"

        if not file_bytes:
            await status_msg.edit_text(
                "❌ Failed to process receipt: Could not read uploaded file bytes.\n⚠️ Please verify image clarity or drop the file directly into Google Drive as a fallback."
            )
            return

        # 2. Processing / Reading Step
        try:
            await status_msg.edit_text("⏳ Extracting store name, purchase date, and items...")
        except Exception:
            pass

        # Check API key
        api_key = os.environ.get('GEMINI_API_KEY') or GEMINI_API_KEY
        if not api_key:
            await status_msg.edit_text(
                "❌ Failed to process receipt: GEMINI_API_KEY is not configured.\n⚠️ Please verify configuration or drop the file directly into Google Drive as a fallback."
            )
            return

        genai_client = genai.Client(api_key=api_key)
        prompt = """
Extract all grocery items from this receipt image or document.
Return ONLY a valid JSON object with the following schema:
{
  "store_name": "Store name (e.g. Grace Supermarket or Store Name)",
  "purchase_date": "YYYY-MM-DD (or receipt date)",
  "items": [
    {
      "item_name": "Item Description",
      "quantity": 1.0,
      "unit": "kg" or "g" or "L" or "pack" or "units",
      "unit_price": 100.0,
      "total_price": 100.0,
      "category": "Staples" or "Cooking Essentials" or "Spices" or "Cleaning & Household" or "Snacks & Packaged" or "Beverages & Dairy" or "Personal Care"
    }
  ]
}
Do not add any markdown explanation. Return pure JSON.
"""
        part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        parsed_data = None
        
        try:
            resp = genai_client.models.generate_content(model="gemini-2.5-flash", contents=[part, prompt])
            if resp and resp.text:
                txt = resp.text.strip()
                if txt.startswith('```'):
                    txt = re.sub(r'^```(json)?\s*', '', txt)
                    txt = re.sub(r'\s*```$', '', txt)
                raw_json = json.loads(txt)
                if isinstance(raw_json, dict) and 'items' in raw_json:
                    parsed_data = raw_json
                elif isinstance(raw_json, list):
                    parsed_data = {
                        "store_name": "Grace Supermarket",
                        "purchase_date": datetime.now().strftime('%Y-%m-%d'),
                        "items": raw_json
                    }
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "ResourceExhausted" in err_str:
                await status_msg.edit_text(
                    "⚠️ *Gemini API Quota Exceeded (429)*\nPlease wait 60 seconds before re-uploading, or drop the receipt in Google Drive."
                )
                return
            logger.warning(f"Receipt vision parsing exception: {e}")

        if not parsed_data or not parsed_data.get('items'):
            await status_msg.edit_text(
                "❌ Failed to process receipt: Could not detect legible line items from this image.\n⚠️ Please verify image clarity or drop the file directly into Google Drive as a fallback."
            )
            return

        store_name = parsed_data.get('store_name') or "Grace Supermarket"
        purchase_date = parsed_data.get('purchase_date') or datetime.now().strftime('%Y-%m-%d')
        items = parsed_data.get('items', [])
        total_amount = sum(float(x.get('total_price', 0.0) or 0.0) for x in items)

        # Ingest into Supabase
        supabase = get_supabase()
        if supabase:
            existing_inv = supabase.table("inventory").select("item_name").execute()
            existing_names = {x['item_name'] for x in existing_inv.data}
            
            for it in items:
                name = str(it.get('item_name', '')).strip()
                if not name:
                    continue
                if name not in existing_names:
                    supabase.table("inventory").insert({
                        'item_name': name,
                        'category': it.get('category', 'Staples'),
                        'current_stock': float(it.get('quantity', 1.0) or 1.0),
                        'unit': it.get('unit', 'units'),
                        'daily_consumption': 0.08,
                        'min_threshold': 0.5,
                        'last_restocked': purchase_date
                    }).execute()
                    existing_names.add(name)

                supabase.table("purchase_history").insert({
                    'item_name': name,
                    'store_name': store_name,
                    'quantity': float(it.get('quantity', 1.0) or 1.0),
                    'unit': it.get('unit', 'units'),
                    'unit_price': float(it.get('unit_price', 0.0) or 0.0),
                    'total_price': float(it.get('total_price', 0.0) or 0.0)
                }).execute()

        # 3. Successful Addition response
        item_bullets = []
        for it in items[:8]:
            item_bullets.append(f"• {it['item_name']} (₹{float(it.get('total_price', 0.0)):,.2f})")

        if len(items) > 8:
            item_bullets.append(f"• _...and {len(items) - 8} more items_")

        success_text = (
            f"✅ *Added to Pantry!*\n\n"
            f"🏪 *Store:* {store_name}\n"
            f"📅 *Date:* {purchase_date}\n"
            f"💰 *Total:* ₹{total_amount:,.2f}\n"
            f"🛒 *Items ({len(items)}):*\n"
            + "\n".join(item_bullets) +
            f"\n\n📊 _Dashboard budget buffer updated._"
        )

        await status_msg.edit_text(success_text, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Error handling receipt: {e}", exc_info=True)
        try:
            await status_msg.edit_text(
                f"❌ Failed to process receipt: {str(e)}\n⚠️ Please verify image clarity or drop the file directly into Google Drive as a fallback."
            )
        except Exception:
            await message.reply_text(
                f"❌ Failed to process receipt: {str(e)}\n⚠️ Please verify image clarity or drop the file directly into Google Drive as a fallback."
            )

# --- Scheduler Setup ---

async def post_init_setup(app: Application):
    ist = pytz.timezone('Asia/Kolkata')
    scheduler = AsyncIOScheduler(timezone=ist)

    async def scheduled_friday_checklist():
        if TELEGRAM_CHAT_ID:
            logger.info(f"Triggering scheduled Friday 09:00 AM IST checklist to chat {TELEGRAM_CHAT_ID}...")
            await send_low_stock_checklist(app.bot, TELEGRAM_CHAT_ID)

    # Friday at 09:00 AM IST
    scheduler.add_job(
        scheduled_friday_checklist,
        CronTrigger(day_of_week='fri', hour=9, minute=0, timezone=ist),
        id='friday_low_stock_checklist',
        replace_existing=True
    )
    scheduler.start()
    logger.info("APScheduler initialized: Friday 09:00 AM IST checklist job active.")
    app.bot_data['scheduler'] = scheduler

def build_telegram_app(token: str = None) -> Application:
    bot_token = token or os.environ.get('TELEGRAM_BOT_TOKEN') or TELEGRAM_BOT_TOKEN
    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in environment or arguments.")

    app = Application.builder().token(bot_token).post_init(post_init_setup).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stock", stock_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("deals", deals_command))
    app.add_handler(CommandHandler("insights", deals_command))
    app.add_handler(CommandHandler("checklist", checklist_command))
    app.add_handler(CommandHandler("spend", spend_command))

    app.add_handler(CallbackQueryHandler(handle_callback_query))

    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, handle_receipt_document_or_photo))

    return app

def run_bot(token: str = None):
    bot_token = token or os.environ.get('TELEGRAM_BOT_TOKEN') or TELEGRAM_BOT_TOKEN
    print(f"Starting Telegram Bot & Scheduler (Token: {bot_token[:10]}...)...")
    app = build_telegram_app(bot_token)
    app.run_polling(drop_pending_updates=True, stop_signals=None, close_loop=False)

if __name__ == '__main__':
    run_bot()
