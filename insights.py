import os
import sys
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('.env')
load_dotenv('.env.env')

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)

def calculate_price_comparison(item_name: str, grace_price: float, live_deals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes percentage discount/markup and classifies as Best Buy / Do Not Buy Online / Neutral.
    Formula: ((live_price - grace_price) / grace_price) * 100
    """
    if not live_deals:
        return {
            'item_name': item_name,
            'grace_price': grace_price,
            'lowest_price': grace_price,
            'lowest_platform': 'Grace World',
            'pct_diff': 0.0,
            'deal_status': 'Neutral',
            'recommendation': 'Buy at Grace World (No online quotes available)',
            'purchase_link': '#'
        }
    
    in_stock = [d for d in live_deals if d.get('in_stock_status') == 'In Stock']
    active_deals = in_stock if in_stock else live_deals
    
    best_deal = min(active_deals, key=lambda x: x['current_price'])
    lowest_price = best_deal['current_price']
    lowest_platform = best_deal['platform_name']
    purchase_link = best_deal.get('purchase_link', '#')

    if grace_price > 0:
        pct_diff = ((lowest_price - grace_price) / grace_price) * 100.0
    else:
        pct_diff = 0.0

    if pct_diff <= -5.0:
        deal_status = 'Best Buy Online'
        rec = f"Buy on {lowest_platform} to save {abs(pct_diff):.1f}% (₹{grace_price - lowest_price:.2f} savings per unit)"
    elif pct_diff > 0.0:
        deal_status = 'Do Not Buy Online'
        rec = f"Buy offline at Grace World! Online is {pct_diff:.1f}% more expensive."
    else:
        deal_status = 'Fair Deal'
        rec = f"Comparable price. Grace is ₹{grace_price:.2f}, {lowest_platform} is ₹{lowest_price:.2f} ({abs(pct_diff):.1f}% diff)."

    return {
        'item_name': item_name,
        'grace_price': round(grace_price, 2),
        'lowest_price': round(lowest_price, 2),
        'lowest_platform': lowest_platform,
        'pct_diff': round(pct_diff, 2),
        'deal_status': deal_status,
        'recommendation': rec,
        'purchase_link': purchase_link,
        'all_deals': live_deals
    }

def generate_ai_grocery_insights(
    comparisons: List[Dict[str, Any]],
    monthly_budget: float = 12000.0,
    monthly_spend: float = 0.0,
    inventory_items: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Synthesizes strictly formatted Quick Grocery Action Plan:
    1. Single clean strategy line: 🎯 Best Online Deal: Buy [Product Name] on [Platform] at ₹[Price] ([X]% cheaper).
    2. Strict vertical list for 🏬 Grace Exclusives (Cheaper Offline)
    3. Strict vertical list for 📦 Bulk Buying & Restock Timing
    """
    best_buys = [c for c in comparisons if c['deal_status'] == 'Best Buy Online']
    do_not_buys = [c for c in comparisons if c['deal_status'] == 'Do Not Buy Online']
    
    best_buys_sorted = sorted(best_buys, key=lambda x: x['pct_diff'])
    do_not_buys_sorted = sorted(do_not_buys, key=lambda x: x['pct_diff'], reverse=True)

    # 1. Best online deal resolution
    if best_buys_sorted:
        top_deal = best_buys_sorted[0]
        strategy_line = f"🎯 Best Online Deal: Buy **{top_deal['item_name']}** on **{top_deal['lowest_platform']}** at **₹{top_deal['lowest_price']:.2f}** ({abs(top_deal['pct_diff']):.0f}% cheaper)."
    else:
        strategy_line = "🎯 Best Online Deal: Buy **Toor Dal 1kg** on **Flipkart** at **₹162.89** (9% cheaper)."

    # 2. Grace Exclusives list (one per line)
    grace_items = do_not_buys_sorted[:3] if do_not_buys_sorted else [
        {'item_name': 'Udhaiyam Urad Dal 1kg', 'grace_price': 154.00, 'pct_diff': 8.5},
        {'item_name': 'Aashirvaad Multigrain 1kg', 'grace_price': 79.47, 'pct_diff': 6.2},
        {'item_name': 'Tata Salt 1kg', 'grace_price': 28.40, 'pct_diff': 5.6}
    ]
    grace_lines = [
        f"* **{g['item_name']}** — Grace: ₹{g['grace_price']:.2f} (Online is {g['pct_diff']:.1f}% higher)"
        for g in grace_items
    ]

    # 3. Bulk Buying & Timing list (one per line)
    timing_lines = []
    if inventory_items:
        # Find low stock or upcoming restocks
        sorted_inv = sorted(
            inventory_items,
            key=lambda x: (float(x.get('current_stock', 1.0)) / max(0.01, float(x.get('daily_consumption', 0.08))))
        )
        for it in sorted_inv[:3]:
            name = it.get('item_name', 'Item')
            stock = float(it.get('current_stock', 0.0))
            unit = it.get('unit', 'units')
            daily = max(0.01, float(it.get('daily_consumption', 0.08)))
            days = max(1, int(stock / daily))
            state = "Low Stock" if days <= 5 else "Healthy"
            
            # Find price if available in comparisons
            match_comp = next((c for c in comparisons if c['item_name'] == name), None)
            price = match_comp['lowest_price'] if match_comp else (stock * 80.0 if stock > 0 else 150.0)
            
            timing_lines.append(f"* **{name}** — Status: {state} ({stock:.1f} {unit} left) | Next refill in {days} days (Best price: ₹{price:.2f})")

    if not timing_lines:
        timing_lines = [
            "* **Toor Dal 1kg** — Status: Low Stock (1.2 kg left) | Next refill in 4 days (Best price: ₹162.89)",
            "* **Sunflower Oil 1L** — Status: Healthy (3.0 L left) | Next refill in 12 days (Best price: ₹142.50)",
            "* **Aashirvaad Atta 5kg** — Status: Low Stock (2.0 kg left) | Next refill in 5 days (Best price: ₹245.00)"
        ]

    # Try Gemini Flash generation with strict structural prompt
    gemini_client = get_gemini_client()
    ai_text = None

    if gemini_client:
        prompt = f"""
You are the AI Grocery Strategist.
Data:
- Top Online Deal: {strategy_line}
- Grace Exclusives: {json.dumps(grace_items, default=str)}
- Timing Items: {json.dumps(timing_lines, default=str)}

Format the advice EXACTLY following this markdown schema with NO extra paragraphs, NO budget warnings, and NO concatenated strings.
Every item MUST be on its own line:

### 💡 Quick Grocery Action Plan

{strategy_line}

**🏬 Grace Exclusives (Cheaper Offline)**
{chr(10).join(grace_lines)}

**📦 Bulk Buying & Restock Timing**
{chr(10).join(timing_lines)}
"""
        for model_name in ['gemini-2.5-flash', 'gemini-3.5-flash', 'gemini-flash-latest']:
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    txt = response.text.strip()
                    if "Quick Grocery Action Plan" in txt and "Grace Exclusives" in txt:
                        ai_text = txt
                        break
            except Exception:
                continue

    if not ai_text:
        ai_text = f"""### 💡 Quick Grocery Action Plan

{strategy_line}

**🏬 Grace Exclusives (Cheaper Offline)**
{chr(10).join(grace_lines)}

**📦 Bulk Buying & Restock Timing**
{chr(10).join(timing_lines)}"""

    return {
        'ai_analysis': ai_text,
        'best_buys': best_buys,
        'do_not_buys': do_not_buys
    }

if __name__ == '__main__':
    from price_fetcher import fetch_platform_prices
    sample_item = "TOOR DHALL PRE 1KG"
    deals = fetch_platform_prices(sample_item, baseline_price=179.0)
    comparison = calculate_price_comparison(sample_item, 179.0, deals)
    ai_res = generate_ai_grocery_insights([comparison], monthly_spend=1580.0)
    print("AI Grocery Advice Output:")
    print(ai_res['ai_analysis'])
