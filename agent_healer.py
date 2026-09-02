import os
import sys

# Ensure current working directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import time
import json
import logging
import threading
import traceback
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv('.env')
load_dotenv('.env.env')

logger = logging.getLogger("AgentHealer")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Configuration Resolver ---

def get_secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is not None:
                if isinstance(val, dict) or hasattr(val, "items"):
                    return json.dumps(dict(val))
                return str(val).strip()
    except Exception:
        pass
    
    val = os.environ.get(key)
    if val:
        return val.strip()
    return default

# --- Global Health & Remediation Registry ---

class SystemHealthRegistry:
    def __init__(self, max_log_size: int = 10):
        self.lock = threading.Lock()
        self.supabase_status: Dict[str, Any] = {"status": "INITIALIZING", "message": "Probing...", "latency_ms": 0}
        self.telegram_status: Dict[str, Any] = {"status": "INITIALIZING", "message": "Starting...", "retries": 0}
        self.drive_status: Dict[str, Any] = {"status": "INITIALIZING", "message": "Evaluating...", "last_poll": None}
        self.llm_status: Dict[str, Any] = {"status": "INITIALIZING", "message": "Checking configuration...", "active_model": "gemini-3.6-flash"}
        self.remediation_log: deque = deque(maxlen=max_log_size)
        self.error_log: deque = deque(maxlen=max_log_size)
        self.supervisor_alive: bool = False
        self.telegram_worker_handle: Optional[threading.Thread] = None
        self.drive_worker_handle: Optional[threading.Thread] = None
        self.telegram_retry_count: int = 0
        self.max_telegram_retries: int = 3
        self.last_drive_auth_check: float = 0
        self.drive_auth_cached_result: Optional[bool] = None

    def log_remediation(self, component: str, action: str, result: str):
        with self.lock:
            entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "component": component,
                "action": action,
                "result": result
            }
            self.remediation_log.appendleft(entry)
            logger.info(f"[Self-Healing] [{component}] {action} -> {result}")

    def log_error(self, component: str, error_msg: str, tb: str = ""):
        with self.lock:
            entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "component": component,
                "error": error_msg,
                "traceback": tb
            }
            self.error_log.appendleft(entry)
            logger.error(f"[Fault Detected] [{component}]: {error_msg}")

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "supabase": dict(self.supabase_status),
                "telegram": dict(self.telegram_status),
                "drive": dict(self.drive_status),
                "llm": dict(self.llm_status),
                "remediations": list(self.remediation_log),
                "errors": list(self.error_log),
                "supervisor_alive": self.supervisor_alive
            }

HEALTH_REGISTRY = SystemHealthRegistry()

# --- Component Probes (Lightweight, Non-Intrusive, No AI Polling) ---

def probe_supabase_db():
    """Lightweight probe verifying Supabase connectivity (SELECT id LIMIT 1) every 60s."""
    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_KEY")
    
    if not url or not key:
        missing = []
        if not url: missing.append("SUPABASE_URL")
        if not key: missing.append("SUPABASE_KEY")
        HEALTH_REGISTRY.supabase_status = {
            "status": "OFFLINE",
            "message": f"Missing credentials: {', '.join(missing)}",
            "latency_ms": 0
        }
        return

    start_t = time.time()
    try:
        from supabase import create_client
        client = create_client(url, key)
        res = client.table("inventory").select("id").limit(1).execute()
        latency = int((time.time() - start_t) * 1000)
        HEALTH_REGISTRY.supabase_status = {
            "status": "HEALTHY",
            "message": "Connected & responsive",
            "latency_ms": latency
        }
    except Exception as e:
        latency = int((time.time() - start_t) * 1000)
        err_str = str(e)
        if "401" in err_str or "Invalid API key" in err_str:
            diag = "Authentication failed (401). Check SUPABASE_KEY."
        elif "timeout" in err_str.lower():
            diag = "Connection timed out to Supabase host."
        else:
            diag = f"Query error: {err_str[:80]}"
            
        HEALTH_REGISTRY.supabase_status = {
            "status": "DEGRADED",
            "message": diag,
            "latency_ms": latency
        }
        HEALTH_REGISTRY.log_error("Supabase", diag, traceback.format_exc())

def probe_google_drive_auth() -> bool:
    """Verifies Google Drive authentication without noisy logs on failure."""
    now = time.time()
    if HEALTH_REGISTRY.drive_auth_cached_result is not None and (now - HEALTH_REGISTRY.last_drive_auth_check < 300):
        return HEALTH_REGISTRY.drive_auth_cached_result

    HEALTH_REGISTRY.last_drive_auth_check = now
    try:
        from drive_watcher import get_drive_service
        service = get_drive_service()
        if service is not None:
            HEALTH_REGISTRY.drive_status = {
                "status": "HEALTHY",
                "message": "OAuth/Service Account Authenticated",
                "last_poll": datetime.now().strftime("%H:%M:%S")
            }
            HEALTH_REGISTRY.drive_auth_cached_result = True
            return True
        else:
            HEALTH_REGISTRY.drive_status = {
                "status": "STANDBY",
                "message": "Standby (No Service Account provided)",
                "last_poll": datetime.now().strftime("%H:%M:%S")
            }
            HEALTH_REGISTRY.drive_auth_cached_result = False
            return False
    except Exception as e:
        HEALTH_REGISTRY.drive_status = {
            "status": "STANDBY",
            "message": f"Standby: {str(e)[:60]}",
            "last_poll": datetime.now().strftime("%H:%M:%S")
        }
        HEALTH_REGISTRY.drive_auth_cached_result = False
        return False

def probe_llm_pipeline():
    """Validates Gemini configuration in-memory without making API calls on a timer."""
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        HEALTH_REGISTRY.llm_status = {
            "status": "OFFLINE",
            "message": "GEMINI_API_KEY missing",
            "active_model": "gemini-3.6-flash"
        }
    else:
        HEALTH_REGISTRY.llm_status = {
            "status": "HEALTHY",
            "message": "Configured & Ready",
            "active_model": "gemini-3.6-flash"
        }

def probe_and_heal_telegram_bot(restart_callback=None):
    """Monitors Telegram Bot thread and auto-restarts if crashed."""
    token = get_secret("TELEGRAM_BOT_TOKEN")
    if not token:
        HEALTH_REGISTRY.telegram_status = {
            "status": "OFFLINE",
            "message": "TELEGRAM_BOT_TOKEN not configured",
            "retries": 0
        }
        return

    tg_thread = HEALTH_REGISTRY.telegram_worker_handle
    if tg_thread is not None and tg_thread.is_alive():
        HEALTH_REGISTRY.telegram_status = {
            "status": "HEALTHY",
            "message": "Polling loop unblocked & active (@Grocery6EBot)",
            "retries": HEALTH_REGISTRY.telegram_retry_count
        }
    else:
        # Bot is down or crashed
        if HEALTH_REGISTRY.telegram_retry_count < HEALTH_REGISTRY.max_telegram_retries and restart_callback:
            HEALTH_REGISTRY.telegram_retry_count += 1
            backoff_delay = 2 ** HEALTH_REGISTRY.telegram_retry_count
            HEALTH_REGISTRY.log_remediation(
                "TelegramBot",
                f"Thread crashed. Executing auto-restart attempt #{HEALTH_REGISTRY.telegram_retry_count} (backoff {backoff_delay}s)...",
                "Initiating restart"
            )
            time.sleep(backoff_delay)
            try:
                new_th = restart_callback()
                HEALTH_REGISTRY.telegram_worker_handle = new_th
                HEALTH_REGISTRY.telegram_status = {
                    "status": "HEALTHY",
                    "message": f"Auto-recovered (Restart #{HEALTH_REGISTRY.telegram_retry_count})",
                    "retries": HEALTH_REGISTRY.telegram_retry_count
                }
                HEALTH_REGISTRY.log_remediation("TelegramBot", "Restart successful", "HEALTHY")
            except Exception as e:
                HEALTH_REGISTRY.log_error("TelegramBot", f"Auto-restart failed: {e}", traceback.format_exc())
                HEALTH_REGISTRY.telegram_status = {
                    "status": "DEGRADED",
                    "message": f"Restart failed: {str(e)[:50]}",
                    "retries": HEALTH_REGISTRY.telegram_retry_count
                }
        else:
            HEALTH_REGISTRY.telegram_status = {
                "status": "OFFLINE",
                "message": f"Offline ({HEALTH_REGISTRY.telegram_retry_count} retries). Use Force Self-Repair.",
                "retries": HEALTH_REGISTRY.telegram_retry_count
            }

# --- Background Supervisor Loop ---

_SUPERVISOR_THREAD: Optional[threading.Thread] = None
_RESTART_BOT_HOOK = None
_RESTART_DRIVE_HOOK = None

def _supervisor_daemon_loop():
    """Supervisor thread running periodic probes (no Gemini API calls on a timer)."""
    HEALTH_REGISTRY.supervisor_alive = True
    logger.info("🛡️ Autonomous Self-Healing SRE Supervisor Thread Started.")
    
    while True:
        try:
            # 1. Probe Supabase DB (every 60s)
            probe_supabase_db()
            
            # 2. Probe Google Drive Auth (cached 5 min)
            probe_google_drive_auth()
            
            # 3. Probe and Heal Telegram Bot
            probe_and_heal_telegram_bot(_RESTART_BOT_HOOK)
            
            # 4. Check LLM Configuration (in-memory, 0 quota used)
            probe_llm_pipeline()
            
        except Exception as e:
            HEALTH_REGISTRY.log_error("Supervisor", f"Supervisor loop tick exception: {e}", traceback.format_exc())

        time.sleep(60)

def register_worker_threads(tg_thread: threading.Thread = None, drive_thread: threading.Thread = None, bot_restart_fn=None, drive_restart_fn=None):
    """Registers background worker handles for SRE supervision."""
    global _RESTART_BOT_HOOK, _RESTART_DRIVE_HOOK
    if tg_thread is not None:
        HEALTH_REGISTRY.telegram_worker_handle = tg_thread
    if drive_thread is not None:
        HEALTH_REGISTRY.drive_worker_handle = drive_thread
    if bot_restart_fn is not None:
        _RESTART_BOT_HOOK = bot_restart_fn
    if drive_restart_fn is not None:
        _RESTART_DRIVE_HOOK = drive_restart_fn

def start_healer_supervisor(tg_thread: threading.Thread = None, drive_thread: threading.Thread = None, bot_restart_fn=None, drive_restart_fn=None):
    """Starts the supervisor thread if not already running."""
    global _SUPERVISOR_THREAD
    register_worker_threads(tg_thread, drive_thread, bot_restart_fn, drive_restart_fn)

    if _SUPERVISOR_THREAD is None or not _SUPERVISOR_THREAD.is_alive():
        _SUPERVISOR_THREAD = threading.Thread(
            target=_supervisor_daemon_loop,
            daemon=True,
            name="SRE_AgentHealerSupervisor"
        )
        _SUPERVISOR_THREAD.start()

def get_health_summary() -> Dict[str, Any]:
    """Retrieves current live health summary."""
    return HEALTH_REGISTRY.get_summary()

def force_self_repair() -> Dict[str, Any]:
    """
    Clears crashed loops, re-reads configuration keys from secrets/env,
    re-runs all probes, and restarts worker threads.
    """
    logger.info("🔧 Force Self-Repair triggered!")
    HEALTH_REGISTRY.telegram_retry_count = 0
    HEALTH_REGISTRY.last_drive_auth_check = 0
    HEALTH_REGISTRY.drive_auth_cached_result = None
    
    probe_supabase_db()
    probe_google_drive_auth()
    probe_llm_pipeline()

    if _RESTART_BOT_HOOK:
        try:
            new_tg = _RESTART_BOT_HOOK()
            HEALTH_REGISTRY.telegram_worker_handle = new_tg
            HEALTH_REGISTRY.log_remediation("TelegramBot", "Force repair restart", "HEALTHY")
        except Exception as e:
            HEALTH_REGISTRY.log_error("TelegramBot", f"Force restart failed: {e}", traceback.format_exc())

    if _RESTART_DRIVE_HOOK:
        try:
            new_dr = _RESTART_DRIVE_HOOK()
            HEALTH_REGISTRY.drive_worker_handle = new_dr
            HEALTH_REGISTRY.log_remediation("DriveWatcher", "Force repair restart", "RESTARTED")
        except Exception as e:
            HEALTH_REGISTRY.log_error("DriveWatcher", f"Force restart failed: {e}", traceback.format_exc())

    HEALTH_REGISTRY.log_remediation("System", "Full Self-Repair Cycle Executed", "SUCCESS")
    return HEALTH_REGISTRY.get_summary()

if __name__ == '__main__':
    print("Testing Agent Healer Self-Diagnostic (0 API calls)...")
    probe_supabase_db()
    probe_google_drive_auth()
    probe_llm_pipeline()
    print("Summary:", json.dumps(HEALTH_REGISTRY.get_summary(), indent=2))
