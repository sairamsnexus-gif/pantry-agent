import os
import sys
import time
import subprocess
import threading
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('.env')
load_dotenv('.env.env')

def run_drive_watcher_worker():
    """Background worker for Google Drive Poller"""
    from drive_watcher import run_drive_watcher
    try:
        run_drive_watcher(interval_seconds=60)
    except Exception as e:
        print(f"[Drive Watcher Worker Error]: {e}")

def run_telegram_bot_worker():
    """Background worker for Telegram Bot and APScheduler"""
    from telegram_bot import run_bot
    try:
        run_bot()
    except Exception as e:
        print(f"[Telegram Bot Worker Error]: {e}")

def main():
    print("=" * 65)
    print("🚀 LAUNCHING FAMILY GROCERY INTELLIGENCE SYSTEM")
    print("=" * 65)

    # 1. Start Google Drive Poller Worker
    print("[1/3] Starting Google Drive Poller (Recursive Ingestion)...")
    drive_thread = threading.Thread(target=run_drive_watcher_worker, daemon=True, name="DriveWatcherThread")
    drive_thread.start()
    print("      ✓ Google Drive Poller running in background.")

    # 2. Start Telegram Bot & APScheduler Worker
    print("[2/3] Starting Telegram Bot (@Grocery6EBot & Friday 9AM IST Scheduler)...")
    tg_thread = threading.Thread(target=run_telegram_bot_worker, daemon=True, name="TelegramBotThread")
    tg_thread.start()
    print("      ✓ Telegram Bot & Friday 09:00 AM IST Checklist active.")

    # 3. Launch Streamlit Web Dashboard
    print("[3/3] Launching Streamlit Web Dashboard on Port 8501...")
    port = os.environ.get("PORT", "8501")
    cmd = [
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("\n" + "=" * 65)
    print(f"🌟 PRODUCTION SYSTEM READY!")
    print(f"🌐 Local Web Dashboard URL: http://localhost:{port}")
    print(f"📱 Telegram Bot: Active & listening for receipts (@Grocery6EBot)")
    print(f"📁 Drive Ingestion: Monitoring folder {os.environ.get('DRIVE_FOLDER_ID')}")
    print("=" * 65 + "\n")

    # Run Streamlit foreground process
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
