import os
import sys

# Ensure current working directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import io
import time
import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google import genai
from google.genai import types
from supabase import create_client
import openpyxl

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv('.env')
load_dotenv('.env.env')

logger = logging.getLogger("DriveWatcher")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
DRIVE_FOLDER_ID = os.environ.get('DRIVE_FOLDER_ID', '1CofRa3fSzj8OEE28OvZHefrffRuMM6cN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL_NAME = "gemini-2.5-flash"
DB_FILE = os.path.join(os.path.dirname(__file__), 'processed_files.db')

_NO_CREDS_WARNED = False
_LAST_429_DRIVE = 0.0

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            file_id TEXT PRIMARY KEY,
            file_name TEXT,
            mime_type TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT,
            items_extracted INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_drive_service():
    """
    Resolves Google Drive API service using:
    1. Local service_account.json / service_account.json.json file
    2. st.secrets["GCP_SERVICE_ACCOUNT"] (as dict or JSON string)
    3. os.environ["GCP_SERVICE_ACCOUNT"] / os.environ["SERVICE_ACCOUNT_JSON"]
    Returns Google Drive service or None if credentials are not configured.
    """
    for sa_path in ['service_account.json', 'service_account.json.json']:
        if os.path.exists(sa_path):
            try:
                creds = service_account.Credentials.from_service_account_file(
                    sa_path,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )
                return build('drive', 'v3', credentials=creds)
            except Exception as e:
                logger.error(f"Error loading credentials from {sa_path}: {e}")

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for sec_key in ["GCP_SERVICE_ACCOUNT", "SERVICE_ACCOUNT_JSON", "gcp_service_account"]:
                if sec_key in st.secrets:
                    val = st.secrets[sec_key]
                    if isinstance(val, dict) or hasattr(val, "items"):
                        info = dict(val)
                    elif isinstance(val, str) and val.strip().startswith("{"):
                        info = json.loads(val.strip())
                    else:
                        continue
                    creds = service_account.Credentials.from_service_account_info(
                        info,
                        scopes=['https://www.googleapis.com/auth/drive.readonly']
                    )
                    return build('drive', 'v3', credentials=creds)
    except Exception:
        pass

    for env_key in ['GCP_SERVICE_ACCOUNT', 'SERVICE_ACCOUNT_JSON']:
        val = os.environ.get(env_key)
        if val:
            try:
                info = json.loads(val) if isinstance(val, str) else dict(val)
                creds = service_account.Credentials.from_service_account_info(
                    info,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )
                return build('drive', 'v3', credentials=creds)
            except Exception as e:
                logger.error(f"Error parsing environment {env_key}: {e}")

    return None

def is_file_processed(file_id: str) -> bool:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT file_id FROM processed_files WHERE file_id = ?", (file_id,))
    row = cursor.fetchone()
    conn.close()
    return row is not None

def mark_file_processed(file_id: str, file_name: str, mime_type: str, status: str, items_count: int):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO processed_files (file_id, file_name, mime_type, processed_at, status, items_extracted)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
    ''', (file_id, file_name, mime_type, status, items_count))
    conn.commit()
    conn.close()

def list_drive_files_recursive(service, folder_id: str) -> List[dict]:
    """Recursively lists all files within a Google Drive folder."""
    files_list = []
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    
    while True:
        results = service.files().list(
            q=query,
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            pageToken=page_token
        ).execute()
        
        items = results.get('files', [])
        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                files_list.extend(list_drive_files_recursive(service, item['id']))
            else:
                files_list.append(item)
                
        page_token = results.get('nextPageToken')
        if not page_token:
            break
            
    return files_list

def parse_bill_image_or_pdf(file_bytes: bytes, mime_type: str, file_name: str) -> List[dict]:
    """Uses Gemini 2.5 Flash Vision to extract itemized grocery line items with 429 protection."""
    global _LAST_429_DRIVE
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not configured for bill vision parsing.")
        return []

    # 60s cooldown if 429 occurred recently
    if time.time() - _LAST_429_DRIVE < 60:
        logger.info("[DriveWatcher] Waiting for 429 rate limit cooldown (60s)...")
        return []

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = """
Extract all grocery items from this receipt / bill image or PDF.
Return ONLY a valid JSON array of objects with the following schema:
[
  {
    "store_name": "Store name (e.g. GRACE WORLD)",
    "item_name": "Item Description (e.g. TOOR DHALL PRE 1KG)",
    "quantity": 1.0,
    "unit": "kg" or "g" or "L" or "pack" or "units",
    "unit_price": 100.0,
    "total_price": 100.0,
    "category": "Staples" or "Cooking Essentials" or "Spices" or "Cleaning & Household" or "Snacks & Packaged" or "Beverages & Dairy" or "Personal Care"
  }
]
Return RAW JSON only. Do not wrap in markdown or backticks.
"""
    try:
        part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[part, prompt]
        )
        if response and response.text:
            txt = response.text.strip()
            if txt.startswith('```'):
                import re
                txt = re.sub(r'^```(json)?\s*', '', txt)
                txt = re.sub(r'\s*```$', '', txt)
            return json.loads(txt)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "ResourceExhausted" in err_str:
            _LAST_429_DRIVE = time.time()
            logger.warning("[DriveWatcher] Gemini 429 quota reached. Pausing vision parsing for 60s.")
        else:
            logger.error(f"Error parsing bill image: {e}")
        
    return []

def process_file_content(service, file_meta: dict):
    file_id = file_meta['id']
    file_name = file_meta['name']
    mime_type = file_meta['mimeType']
    
    logger.info(f"Processing Drive file: {file_name} ({file_id})")
    
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        
    file_bytes = fh.getvalue()
    extracted_items = []

    if 'spreadsheet' in mime_type or file_name.endswith(('.xlsx', '.xls')):
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
        sheet_name = 'All Consolidated Items' if 'All Consolidated Items' in wb.sheetnames else wb.sheetnames[0]
        ws = wb[sheet_name]
        for r in range(5, ws.max_row + 1):
            store = ws.cell(row=r, column=3).value or 'Grace World'
            item_desc = ws.cell(row=r, column=5).value
            qty = ws.cell(row=r, column=7).value
            unit_rate = ws.cell(row=r, column=9).value
            total_amt = ws.cell(row=r, column=10).value
            if item_desc:
                try:
                    q = float(qty) if qty else 1.0
                    ur = float(unit_rate) if unit_rate else 0.0
                    tot = float(total_amt) if total_amt else (q * ur)
                except:
                    q, ur, tot = 1.0, 0.0, 0.0
                extracted_items.append({
                    'store_name': str(store),
                    'item_name': str(item_desc).strip(),
                    'quantity': q,
                    'unit': 'kg' if 'KG' in str(item_desc).upper() else ('L' if '1L' in str(item_desc).upper() else 'units'),
                    'unit_price': ur,
                    'total_price': tot,
                    'category': 'Staples'
                })
    elif 'image' in mime_type or 'pdf' in mime_type or file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf', '.webp')):
        extracted_items = parse_bill_image_or_pdf(file_bytes, mime_type, file_name)
    else:
        mark_file_processed(file_id, file_name, mime_type, 'SKIPPED_UNSUPPORTED', 0)
        return

    logger.info(f"Extracted {len(extracted_items)} items from {file_name}")

    if extracted_items and SUPABASE_URL and SUPABASE_KEY:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        existing_inv = client.table("inventory").select("item_name").execute()
        existing_names = {x['item_name'] for x in existing_inv.data}
        
        new_inv_items = []
        for it in extracted_items:
            name = it['item_name']
            if name not in existing_names:
                new_inv_items.append({
                    'item_name': name,
                    'category': it.get('category', 'Staples'),
                    'current_stock': it.get('quantity', 1.0),
                    'unit': it.get('unit', 'units'),
                    'daily_consumption': 0.08,
                    'min_threshold': 0.5,
                    'last_restocked': datetime.now(timezone.utc).strftime('%Y-%m-%d')
                })
                existing_names.add(name)
                
        if new_inv_items:
            for b in range(0, len(new_inv_items), 50):
                client.table("inventory").insert(new_inv_items[b:b+50]).execute()

        purchases = [
            {
                'item_name': it['item_name'],
                'store_name': it.get('store_name', 'Grace World'),
                'quantity': it.get('quantity', 1.0),
                'unit': it.get('unit', 'units'),
                'unit_price': it.get('unit_price', 0.0),
                'total_price': it.get('total_price', 0.0)
            }
            for it in extracted_items
        ]
        
        for b in range(0, len(purchases), 50):
            client.table("purchase_history").insert(purchases[b:b+50]).execute()

    mark_file_processed(file_id, file_name, mime_type, 'SUCCESS', len(extracted_items))
    logger.info(f"Drive file {file_name} successfully ingested into Supabase!")

def poll_drive_folder():
    """Single polling cycle checking Google Drive folder for new bills."""
    global _NO_CREDS_WARNED
    try:
        service = get_drive_service()
        if not service:
            if not _NO_CREDS_WARNED:
                logger.warning("[Google Drive Poller] No Service Account credentials configured. Drive watcher is in standby mode.")
                _NO_CREDS_WARNED = True
            return

        files = list_drive_files_recursive(service, DRIVE_FOLDER_ID)
        for f in files:
            if not is_file_processed(f['id']):
                process_file_content(service, f)
    except Exception as e:
        logger.error(f"Drive poller error: {e}")

def run_drive_watcher(interval_seconds: int = 60):
    """Continuous background loop monitoring Google Drive with minimum 60s interval."""
    interval = max(60, interval_seconds)
    init_db()
    while True:
        poll_drive_folder()
        time.sleep(interval)

if __name__ == '__main__':
    poll_drive_folder()
