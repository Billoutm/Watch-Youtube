import os
import re
import json
import time
import random
import shutil
import hashlib
import subprocess
import traceback
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
import customtkinter as ctk
from tkinter import messagebox
import threading

load_dotenv()
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DATABASE_ID = os.getenv("DATABASE_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

STARTUP_ERROR = None
if not NOTION_API_KEY or not DATABASE_ID:
    STARTUP_ERROR = "Vérifie que NOTION_API_KEY et DATABASE_ID sont définis dans le fichier .env."
elif not GEMINI_API_KEY:
    STARTUP_ERROR = "Ajoute GEMINI_API_KEY dans ton fichier .env (gratuit sur https://aistudio.google.com/apikey)."

# ============================================================
# CONSTANTES
# ============================================================

GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODELS = ["gemini-3.7-flash"]  # essayé si GEMINI_MODEL reste surchargé après toutes les tentatives

# Durée des tranches et résolution : adaptées selon la disponibilité de sous-titres officiels
# (si on les a, Gemini n'a plus besoin de deviner les paroles -> on peut baisser la résolution
# demandée et allonger les tranches, ce qui réduit la conso de quota).
CHUNK_DURATION_WITH_CAPTIONS = 30 * 60
CHUNK_DURATION_NO_CAPTIONS = 20 * 60
MEDIA_RESOLUTION_WITH_CAPTIONS = "MEDIA_RESOLUTION_MEDIUM"
MEDIA_RESOLUTION_NO_CAPTIONS = "MEDIA_RESOLUTION_HIGH"

MAX_VIDEO_HEIGHT = 720
FILE_READY_TIMEOUT = 300
GEMINI_MAX_RETRIES = 2
DELAY_BETWEEN_PAGES = 3
MAX_FAILED_ATTEMPTS = 3
MAX_PAGES_FOR_MATCHING = 5  # nb de pages existantes envoyées à Gemini pour la comparaison de sujet

OVERLOAD_MAX_ATTEMPTS = 3
OVERLOAD_BASE_DELAY = 15
OVERLOAD_MAX_DELAY = 240

DEFAULT_CATEGORY = "Divers"
CACHE_DIR = "video_cache"
MIGRATION_FAILURES_FILE = "migration_failures.json"  # uniquement les compteurs d'échecs (pas le statut "fait")
CHANNELS_DATABASE_ID = os.getenv("CHANNELS_DATABASE_ID")  # optionnel : onglet "Historique chaînes"
BACKFILL_STATE_FILE = "channel_backfill_state.json"
LOG_FILE = "app_log.txt"
LANG_INSTRUCTION = "IMPORTANT : quelle que soit la langue parlée dans la vidéo (français, anglais, etc.), tu dois TOUJOURS répondre en français."

# Propriétés Notion
PROP_TITLE = "Nom"
PROP_PRICE = "Prix"
PROP_SOURCE_1 = "Sources"
PROP_SOURCE_2 = "2 sources (facultatives)"
PROP_TYPE = "types"
PROP_GEMINI = "Gemini"  # checkbox : coché = déjà traité/mis à jour par Gemini

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if not STARTUP_ERROR else None
AFC_DISABLED = types.AutomaticFunctionCallingConfig(disable=True) if not STARTUP_ERROR else None


# ============================================================
# EXCEPTIONS DE CONTRÔLE
# ============================================================

class QuotaExceededError(Exception):
    """Quota/rate-limit Gemini atteint : on s'arrête, le travail déjà fait est préservé sur disque."""
    pass


class ServiceOverloadedError(Exception):
    """Gemini reste surchargé (503...) après plusieurs tentatives espacées : on se met en pause."""
    pass


class StopRequested(Exception):
    """L'utilisateur a cliqué sur Arrêter pendant une attente ou un traitement."""
    pass


# ============================================================
# LOG (fichier + callback GUI)
# ============================================================

def write_log_file(msg):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def make_logger(textbox, root):
    def log_fn(msg):
        write_log_file(msg)
        textbox.configure(state="normal")
        textbox.insert("end", str(msg) + "\n")
        textbox.see("end")
        textbox.configure(state="disabled")
        root.update()
    return log_fn


# ============================================================
# CLASSIFICATION DES ERREURS + PROTECTION 503 / 429
# ============================================================

QUOTA_KEYWORDS = ["429", "resource_exhausted", "quota", "rate limit", "rate_limit"]
OVERLOAD_CODES = (500, 502, 503, 504)
OVERLOAD_KEYWORDS = ["503", "504", "unavailable", "overloaded", "high demand",
                     "deadline_exceeded", "try again later", "internal error"]


def is_quota_error(err_text):
    err_text = (err_text or "").lower()
    return any(k in err_text for k in QUOTA_KEYWORDS)


def classify_error(exc):
    code = getattr(exc, "code", None)
    text = str(exc).lower()
    if code == 429 or any(k in text for k in QUOTA_KEYWORDS):
        return "quota"
    if code in OVERLOAD_CODES or any(k in text for k in OVERLOAD_KEYWORDS):
        return "overloaded"
    return "other"


def interruptible_sleep(seconds, stop_event):
    end = time.time() + seconds
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            return
        if stop_event and stop_event.is_set():
            raise StopRequested()
        time.sleep(min(1, remaining))


def call_with_overload_protection(func, stop_event, log_fn, description="appel Gemini"):
    for attempt in range(1, OVERLOAD_MAX_ATTEMPTS + 1):
        try:
            return func()
        except Exception as e:
            kind = classify_error(e)
            if kind == "quota":
                raise QuotaExceededError(str(e))
            if kind != "overloaded":
                raise
            if attempt == OVERLOAD_MAX_ATTEMPTS:
                raise ServiceOverloadedError(str(e))
            delay = min(OVERLOAD_BASE_DELAY * (2 ** (attempt - 1)), OVERLOAD_MAX_DELAY)
            delay += random.uniform(0, delay * 0.2)
            log_fn(f"   ⏳ Gemini surchargé pendant « {description} » (essai {attempt}/{OVERLOAD_MAX_ATTEMPTS}). "
                   f"Nouvel essai dans {int(delay)}s...")
            interruptible_sleep(delay, stop_event)


# ============================================================
# VÉRIFICATIONS DE DÉMARRAGE
# ============================================================

def check_dependencies(log_fn):
    missing = [tool for tool in ("ffmpeg", "ffprobe", "yt-dlp") if shutil.which(tool) is None]
    if missing:
        log_fn(f"❌ Outils manquants dans le PATH : {', '.join(missing)}")
        log_fn("💡 Réinstalle-les (ex: après une réinstallation d'OS, ils ne sont plus dans le PATH).")
        return False
    if shutil.which("deno") is None:
        log_fn("⚠️ Deno introuvable (requis par yt-dlp pour contourner les protections YouTube).")
        log_fn("💡 Installe-le : https://deno.land, puis redémarre le terminal.")
    return True


def test_connections(log_fn, stop_event):
    ok = True
    log_fn("🔧 Test de la connexion Notion...")
    try:
        get_data_source_id(DATABASE_ID, log_fn)
        log_fn("   ✅ Notion OK")
    except Exception as e:
        log_fn(f"   ❌ Notion KO : {e}")
        ok = False

    log_fn("🔧 Test de la connexion Gemini...")
    try:
        call_with_overload_protection(lambda: gemini_client.models.list(), stop_event, log_fn, "test de connexion")
        log_fn("   ✅ Gemini OK")
    except ServiceOverloadedError:
        log_fn("   ⚠️ Gemini très chargé en ce moment, le traitement attendra si nécessaire.")
    except QuotaExceededError as e:
        log_fn(f"   ⏳ Quota déjà atteint : {e}")
        ok = False
    except StopRequested:
        ok = False
    except Exception as e:
        log_fn(f"   ❌ Gemini KO : {e}")
        ok = False
    return ok


# ============================================================
# CACHE DISQUE RÉSUMABLE (reprise au niveau de la tranche)
# ============================================================

def get_work_id_for_url(url):
    return "url_" + hashlib.md5(url.encode("utf-8")).hexdigest()[:16]


def work_dir(work_id):
    d = os.path.join(CACHE_DIR, work_id)
    os.makedirs(d, exist_ok=True)
    return d


def state_path(work_id):
    return os.path.join(work_dir(work_id), "state.json")


def load_work_state(work_id):
    p = state_path(work_id)
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {"video_title": None, "description": None, "captions": None, "chunks": {}, "synthesis": None}


def save_work_state(work_id, state):
    with open(state_path(work_id), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def clear_work(work_id, log_fn=None):
    d = os.path.join(CACHE_DIR, work_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        if log_fn:
            log_fn(f"🧹 Cache nettoyé pour {work_id}.")


def clear_all_cache(log_fn=None):
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    if log_fn:
        log_fn("🧹 Tout le cache vidéo a été vidé.")


# ============================================================
# COMPTEURS D'ÉCHECS DE MIGRATION (remplace l'ancien migration_progress.json)
# Le statut "fait" vit maintenant dans Notion (propriété Gemini) ; ce fichier
# ne sert qu'à éviter de retenter indéfiniment une page définitivement cassée.
# ============================================================

def load_failures():
    if os.path.isfile(MIGRATION_FAILURES_FILE):
        try:
            with open(MIGRATION_FAILURES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_failures(failures):
    with open(MIGRATION_FAILURES_FILE, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)


def record_failure(failures, page_id, title, error):
    entry = failures.get(page_id, {"attempts": 0, "title": title})
    entry["attempts"] += 1
    entry["error"] = str(error)
    entry["title"] = title
    failures[page_id] = entry
    save_failures(failures)
    return entry["attempts"]


def clear_failure(failures, page_id):
    if page_id in failures:
        del failures[page_id]
        save_failures(failures)


# ============================================================
# NOTION
# ============================================================

def notion_request(method, url, log_fn, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.setdefault("Authorization", f"Bearer {NOTION_API_KEY}")
    headers.setdefault("Notion-Version", "2025-09-03")
    last_exc = None
    for attempt in range(2):
        try:
            return requests.request(method, url, headers=headers, timeout=30, **kwargs)
        except requests.exceptions.RequestException as e:
            last_exc = e
            log_fn(f"⚠️ Erreur réseau Notion (essai {attempt + 1}/2) : {e}")
            time.sleep(2)
    raise last_exc


def get_data_source_id(database_id, log_fn):
    clean_id = ''.join(c for c in database_id if c in '0123456789abcdefABCDEF-').replace('-', '')
    if len(clean_id) == 32:
        formatted_id = f"{clean_id[:8]}-{clean_id[8:12]}-{clean_id[12:16]}-{clean_id[16:20]}-{clean_id[20:]}"
    else:
        formatted_id = database_id

    response = notion_request("GET", f"https://api.notion.com/v1/databases/{formatted_id}", log_fn)
    response.raise_for_status()
    data_sources = response.json().get("data_sources", [])
    if not data_sources:
        raise ValueError("Aucune data source trouvée dans cette base de données.")
    return data_sources[0]["id"]


def get_existing_categories(data_source_id, log_fn):
    try:
        response = notion_request("GET", f"https://api.notion.com/v1/data_sources/{data_source_id}", log_fn)
        response.raise_for_status()
        types_prop = response.json().get("properties", {}).get(PROP_TYPE, {})
        options = types_prop.get("select", {}).get("options", [])
        return [o["name"] for o in options]
    except Exception as e:
        log_fn(f"⚠️ Impossible de récupérer les catégories existantes : {e}")
        return []


def search_existing_pages(data_source_id, video_type, log_fn):
    payload = {"filter": {"property": PROP_TYPE, "select": {"equals": video_type}}}
    try:
        response = notion_request("POST", f"https://api.notion.com/v1/data_sources/{data_source_id}/query",
                                  log_fn, json=payload)
        response.raise_for_status()
        pages = []
        for page in response.json().get("results", []):
            title_prop = page["properties"].get(PROP_TITLE, {}).get("title", [])
            title = title_prop[0]["text"]["content"] if title_prop else "Sans titre"
            pages.append({"id": page["id"], "title": title})
        return pages
    except Exception as e:
        log_fn(f"⚠️ Erreur lors de la recherche : {e}")
        return []


def find_page_by_source_url(data_source_id, url, log_fn):
    """Vérifie si cette URL exacte est déjà une source d'une page existante (évite de gaspiller
    du quota Gemini sur un doublon évident). Ne couvre que les 2 premières sources ; les sources
    'supplémentaires' (3e et +) ne sont stockées qu'en texte dans le corps de page, pas dans une
    propriété filtrable."""
    payload = {"filter": {"or": [
        {"property": PROP_SOURCE_1, "url": {"equals": url}},
        {"property": PROP_SOURCE_2, "url": {"equals": url}},
    ]}}
    try:
        response = notion_request("POST", f"https://api.notion.com/v1/data_sources/{data_source_id}/query",
                                  log_fn, json=payload)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None
        page = results[0]
        title_prop = page["properties"].get(PROP_TITLE, {}).get("title", [])
        title = title_prop[0]["text"]["content"] if title_prop else "Sans titre"
        return {"id": page["id"], "title": title}
    except Exception as e:
        log_fn(f"⚠️ Erreur vérification doublon : {e}")
        return None


def list_all_pages(data_source_id, log_fn):
    pages = []
    payload = {"page_size": 100}
    while True:
        response = notion_request("POST", f"https://api.notion.com/v1/data_sources/{data_source_id}/query",
                                  log_fn, json=payload)
        response.raise_for_status()
        data = response.json()
        pages.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        payload["start_cursor"] = data["next_cursor"]
    return pages


def extract_page_info(page):
    props = page.get("properties", {})
    title_list = props.get(PROP_TITLE, {}).get("title", [])
    title = title_list[0]["text"]["content"] if title_list else "Sans titre"
    video_url = props.get(PROP_SOURCE_1, {}).get("url")
    type_prop = props.get(PROP_TYPE, {}).get("select")
    video_type = type_prop["name"] if type_prop else DEFAULT_CATEGORY
    is_gemini_done = bool(props.get(PROP_GEMINI, {}).get("checkbox"))
    return title, video_url, video_type, is_gemini_done


def get_page_content(page_id, log_fn):
    try:
        response = notion_request("GET", f"https://api.notion.com/v1/blocks/{page_id}/children", log_fn)
        response.raise_for_status()
        content = ""
        for block in response.json().get("results", []):
            if block["type"] == "paragraph":
                content += "".join(t["text"]["content"] for t in block["paragraph"]["rich_text"]) + "\n"
        return content.strip()
    except Exception as e:
        log_fn(f"⚠️ Erreur lecture page : {e}")
        return ""


def get_page_source_urls(page_id, log_fn):
    """Lit TOUTES les sources connues d'une page : les 2 propriétés url + la section
    'Sources supplémentaires' du corps de page si elle existe."""
    sources = []
    try:
        response = notion_request("GET", f"https://api.notion.com/v1/pages/{page_id}", log_fn)
        response.raise_for_status()
        props = response.json().get("properties", {})
        for prop_name in (PROP_SOURCE_1, PROP_SOURCE_2):
            url = props.get(prop_name, {}).get("url")
            if url:
                sources.append(url)
    except Exception as e:
        log_fn(f"⚠️ Erreur lecture sources (propriétés) : {e}")

    try:
        response = notion_request("GET", f"https://api.notion.com/v1/blocks/{page_id}/children", log_fn)
        response.raise_for_status()
        in_sources_section = False
        for block in response.json().get("results", []):
            if block["type"] == "heading_2":
                heading_text = "".join(t["text"]["content"] for t in block["heading_2"]["rich_text"])
                in_sources_section = "Sources" in heading_text
                continue
            if in_sources_section and block["type"] == "bulleted_list_item":
                for t in block["bulleted_list_item"]["rich_text"]:
                    link = t.get("text", {}).get("link")
                    if link and link.get("url"):
                        sources.append(link["url"])
    except Exception as e:
        log_fn(f"⚠️ Erreur lecture sources (contenu) : {e}")

    # dédoublonnage en gardant l'ordre
    seen = set()
    ordered = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def _domain_label(url):
    m = re.search(r"https?://(?:www\.)?([^/]+)", url or "")
    return m.group(1) if m else "lien"


def _content_blocks(summary, a_retenir, transcription, all_sources, updated=False):
    suffix = " (mis à jour)" if updated else ""
    safe_summary = (summary or "")[:2000]
    safe_a_retenir = (a_retenir or "")[:2000]
    safe_transcription = (transcription or "")[:2000]
    blocks = [
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": f"📝 Synthèse{suffix}"}}]}},
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": safe_summary}}]}},
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": f"⭐ À retenir{suffix}"}}]}},
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": safe_a_retenir}}]}},
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": "🔗 Sources"}}]}},
    ]
    for src in all_sources:
        blocks.append({
            "object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": _domain_label(src), "link": {"url": src}}}]}
        })
    blocks += [
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": f"📄 Transcription complète{suffix}"}}]}},
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": safe_transcription}}]}}
    ]
    return blocks


def replace_page_content(page_id, summary, a_retenir, transcription, all_sources, log_fn, updated=False):
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    try:
        response = notion_request("GET", url, log_fn)
        response.raise_for_status()
        for block in response.json().get("results", []):
            notion_request("DELETE", f"https://api.notion.com/v1/blocks/{block['id']}", log_fn)
    except Exception as e:
        log_fn(f"⚠️ Erreur suppression ancien contenu : {e}")

    try:
        response = notion_request("PATCH", url, log_fn,
                                  json={"children": _content_blocks(summary, a_retenir, transcription, all_sources, updated)})
        response.raise_for_status()
        return True
    except Exception as e:
        log_fn(f"❌ Erreur mise à jour contenu : {e}")
        return False


def update_page_properties(page_id, new_title, new_category, all_sources, log_fn):
    """Met à jour titre, catégorie, les 2 propriétés url (1ère et 2e source), et coche 'Gemini'."""
    properties = {
        PROP_TITLE: {"title": [{"text": {"content": new_title[:100]}}]},
        PROP_TYPE: {"select": {"name": new_category}},
        PROP_GEMINI: {"checkbox": True},
    }
    if len(all_sources) >= 1:
        properties[PROP_SOURCE_1] = {"url": all_sources[0]}
    if len(all_sources) >= 2:
        properties[PROP_SOURCE_2] = {"url": all_sources[1]}

    try:
        response = notion_request("PATCH", f"https://api.notion.com/v1/pages/{page_id}", log_fn, json={"properties": properties})
        response.raise_for_status()
        return True
    except Exception as e:
        log_fn(f"⚠️ Impossible de mettre à jour les propriétés de la page : {e}")
        return False


def send_to_notion(video_url, transcription, summary, title, video_type, data_source_id, a_retenir, log_fn):
    payload = {
        "parent": {"type": "data_source_id", "data_source_id": data_source_id},
        "properties": {
            PROP_TITLE: {"title": [{"text": {"content": title[:100]}}]},
            PROP_PRICE: {"select": {"name": "Gratuite"}},
            PROP_SOURCE_1: {"url": video_url},
            PROP_TYPE: {"select": {"name": video_type}},
            PROP_GEMINI: {"checkbox": True},
        },
        "children": _content_blocks(summary, a_retenir, transcription, [video_url])
    }
    try:
        response = notion_request("POST", "https://api.notion.com/v1/pages", log_fn, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        log_fn(f"❌ Erreur Notion : {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        log_fn(f"❌ Erreur lors de l'envoi à Notion : {e}")
        return None


# ============================================================
# MÉTADONNÉES + SOUS-TITRES OFFICIELS (réduction du coût Gemini)
# ============================================================

def get_video_metadata(url, log_fn):
    """Titre + description via yt-dlp, sans télécharger la vidéo. Gratuit, utile pour toutes
    les plateformes supportées (YouTube, Instagram, TikTok...)."""
    if os.path.isfile(url):
        return {"title": os.path.splitext(os.path.basename(url))[0], "description": ""}
    try:
        result = subprocess.run(["yt-dlp", "-J", "--skip-download", url],
                                capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(result.stdout)
        return {"title": data.get("title", ""), "description": (data.get("description") or "")[:1500]}
    except Exception as e:
        log_fn(f"⚠️ Métadonnées indisponibles : {e}")
        return {"title": "", "description": ""}


def _parse_vtt(content):
    lines = content.splitlines()
    text_lines = []
    cue_re = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->")
    for line in lines:
        line = line.strip()
        if not line or line.upper().startswith("WEBVTT") or line.isdigit() or cue_re.match(line):
            continue
        clean = re.sub(r"<[^>]+>", "", line)
        if clean and (not text_lines or text_lines[-1] != clean):
            text_lines.append(clean)
    return " ".join(text_lines)[:20000]


def try_fetch_captions(work_id, url, log_fn):
    """Récupère les sous-titres OFFICIELS (ou auto-générés par la plateforme) si disponibles.
    Gratuit et quasi instantané (pas de téléchargement vidéo). Renvoie None si indisponibles
    (cas fréquent sur Instagram par exemple)."""
    if os.path.isfile(url):
        return None

    wd = work_dir(work_id)
    out_template = os.path.join(wd, "captions")
    try:
        subprocess.run([
            "yt-dlp", "--skip-download", "--write-subs", "--write-auto-subs",
            "--sub-langs", "fr,fr-FR,en,en-US",
            "--sub-format", "vtt",
            "-o", out_template, url
        ], capture_output=True, text=True, timeout=60)
    except Exception as e:
        log_fn(f"   ℹ️ Pas de sous-titres officiels récupérables : {e}")
        return None

    for fname in os.listdir(wd):
        if fname.startswith("captions") and fname.endswith(".vtt"):
            try:
                with open(os.path.join(wd, fname), "r", encoding="utf-8") as f:
                    text = _parse_vtt(f.read())
                if text:
                    return text
            except Exception:
                continue
    return None


# ============================================================
# TÉLÉCHARGEMENT VIDÉO (mis en cache par work_id)
# ============================================================

def _try_ytdlp_download(url, video_path, cookies_browser=None):
    base_args = ["yt-dlp"]
    if cookies_browser:
        base_args += ["--cookies-from-browser", cookies_browser]
    try:
        result = subprocess.run(base_args + ["--get-title", url], capture_output=True, text=True, check=True)
        video_title = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, None, (e.stderr or str(e))

    try:
        subprocess.run(base_args + [
            "-f", f"bv*[height<={MAX_VIDEO_HEIGHT}][ext=mp4]+ba[ext=m4a]/b[height<={MAX_VIDEO_HEIGHT}][ext=mp4]/best[height<={MAX_VIDEO_HEIGHT}]/best",
            "--merge-output-format", "mp4", "-o", video_path, url
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return False, None, (e.stderr or str(e))

    if not os.path.isfile(video_path) or os.path.getsize(video_path) < 1024:
        return False, None, "Fichier vide ou introuvable après téléchargement."
    return True, video_title, None


def download_video_cached(work_id, url, fallback_title, log_fn):
    video_path = os.path.join(work_dir(work_id), "source.mp4")
    state = load_work_state(work_id)

    if os.path.isfile(video_path) and os.path.getsize(video_path) > 1024:
        log_fn("♻️ Vidéo déjà téléchargée (reprise depuis le cache).")
        return video_path, state.get("video_title") or fallback_title

    if os.path.isfile(url):
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", url,
                "-vf", f"scale=-2:'min({MAX_VIDEO_HEIGHT},ih)'",
                "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
                video_path
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            log_fn(f"❌ Erreur conversion vidéo locale : {e.stderr[-300:] if e.stderr else e}")
            return None, None
        title = os.path.splitext(os.path.basename(url))[0]
        state["video_title"] = title
        save_work_state(work_id, state)
        return video_path, title

    log_fn("⬇️ Téléchargement de la vidéo avec yt-dlp...")
    preferred_browser = os.getenv("YT_DLP_COOKIES_BROWSER")
    attempts = [("sans cookies", None)]
    if preferred_browser:
        attempts.append((f"cookies {preferred_browser}", preferred_browser))
    else:
        attempts.append(("cookies firefox", "firefox"))
        attempts.append(("cookies chrome", "chrome"))

    last_error = None
    for label, browser in attempts:
        log_fn(f"   🔄 Tentative : {label}...")
        success, video_title, error = _try_ytdlp_download(url, video_path, cookies_browser=browser)
        if success:
            log_fn(f"   ✅ Téléchargement réussi ({label}).")
            title = video_title or fallback_title
            state["video_title"] = title
            save_work_state(work_id, state)
            return video_path, title
        last_error = error
        log_fn(f"   ⚠️ Échec ({label}) : {error[-200:] if error else 'erreur inconnue'}")

    log_fn(f"❌ Toutes les tentatives ont échoué. Dernière erreur : {last_error[-300:] if last_error else 'inconnue'}")
    return None, None


def get_video_duration(video_path):
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video_path
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return None


def split_video_cached(work_id, video_path, chunk_duration, log_fn):
    duration = get_video_duration(video_path)
    if duration is None or duration <= chunk_duration:
        return [video_path], duration

    nb_chunks = int(duration // chunk_duration) + (1 if duration % chunk_duration > 0 else 0)
    chunk_paths = []
    wd = work_dir(work_id)

    for i in range(nb_chunks):
        chunk_path = os.path.join(wd, f"chunk_{i + 1}.mp4")
        if os.path.isfile(chunk_path) and os.path.getsize(chunk_path) > 1024:
            chunk_paths.append(chunk_path)
            continue

        start = i * chunk_duration
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-ss", str(start), "-t", str(chunk_duration),
                "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
                "-avoid_negative_ts", "make_zero", chunk_path
            ], check=True, capture_output=True, text=True)
            if os.path.isfile(chunk_path) and os.path.getsize(chunk_path) > 1024:
                chunk_paths.append(chunk_path)
            else:
                log_fn(f"⚠️ Tranche {i + 1} vide ou invalide, ignorée.")
        except subprocess.CalledProcessError as e:
            log_fn(f"⚠️ Erreur découpage tranche {i + 1} : {e.stderr[-300:] if e.stderr else e}")

    return (chunk_paths if chunk_paths else [video_path]), duration


# ============================================================
# GEMINI
# ============================================================

def upload_and_wait(file_path, log_fn, stop_event, timeout=FILE_READY_TIMEOUT):
    log_fn(f"      ⬆️ Upload vers Gemini : {os.path.basename(file_path)}...")
    uploaded = call_with_overload_protection(
        lambda: gemini_client.files.upload(file=file_path), stop_event, log_fn, "upload de la vidéo"
    )
    waited = 0
    while uploaded.state.name == "PROCESSING" and waited < timeout:
        interruptible_sleep(5, stop_event)
        waited += 5
        uploaded = call_with_overload_protection(
            lambda: gemini_client.files.get(name=uploaded.name), stop_event, log_fn, "suivi du traitement du fichier"
        )
    if uploaded.state.name != "ACTIVE":
        log_fn(f"      ❌ Fichier non prêt après {waited}s (état : {uploaded.state.name})")
        return None
    return uploaded


def extract_json(raw_text):
    if not raw_text:
        return None
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def extract_response_text(response):
    if not getattr(response, "candidates", None):
        return None, "Aucune réponse (candidates vide)"
    candidate = response.candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return None, f"Pas de contenu (finish_reason={finish_reason})"
    text = "".join(getattr(p, "text", "") or "" for p in parts)
    if not text.strip():
        return None, f"Contenu vide (finish_reason={finish_reason})"
    return text, None


def call_gemini(contents, log_fn, stop_event, max_output_tokens=2000, temperature=0.3,
                retries=GEMINI_MAX_RETRIES, media_resolution=None):
    config_kwargs = dict(
        response_mime_type="application/json",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        automatic_function_calling=AFC_DISABLED
    )
    if media_resolution:
        config_kwargs["media_resolution"] = getattr(types.MediaResolution, media_resolution)
    config = types.GenerateContentConfig(**config_kwargs)

    models_to_try = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    last_error = None

    for model_name in models_to_try:
        try:
            for attempt in range(1, retries + 2):
                response = call_with_overload_protection(
                    lambda: gemini_client.models.generate_content(model=model_name, contents=contents, config=config),
                    stop_event, log_fn, f"analyse Gemini ({model_name})"
                )
                text, error = extract_response_text(response)
                if error:
                    if is_quota_error(error):
                        raise QuotaExceededError(error)
                    last_error = error
                    interruptible_sleep(2 * attempt, stop_event)
                    continue

                data = extract_json(text)
                if data:
                    return data
                last_error = "JSON introuvable dans la réponse"
                interruptible_sleep(2 * attempt, stop_event)
        except ServiceOverloadedError as e:
            last_error = str(e)
            if model_name != models_to_try[-1]:
                next_model = models_to_try[models_to_try.index(model_name) + 1]
                log_fn(f"      🔀 {model_name} indisponible (surcharge persistante), bascule sur {next_model}...")
                continue
            raise  # dernier modèle de la liste : on laisse l'appelant gérer la pause

    log_fn(f"      ❌ Échec définitif sur tous les modèles essayés ({', '.join(models_to_try)}) : {last_error}")
    return None


def _captions_block(captions_text):
    if not captions_text:
        return ""
    return (f"\nSOUS-TITRES OFFICIELS DISPONIBLES (fais confiance à ce texte pour les paroles exactes, "
            f"concentre-toi sur l'image pour le reste : schémas, démonstrations, texte affiché à l'écran) :\n"
            f"{captions_text[:6000]}\n")


def analyze_video_chunk(uploaded_file, chunk_number, total_chunks, captions_text, media_resolution, log_fn, stop_event):
    is_last = (chunk_number == total_chunks)
    captions_block = _captions_block(captions_text)

    if is_last:
        prompt = f"""Regarde et écoute cette DERNIÈRE partie ({chunk_number}/{total_chunks}) d'une vidéo éducative sur l'audiovisuel.
{LANG_INSTRUCTION}
{captions_block}
C'est la fin : cherche un récapitulatif/une conclusion.
Réponds UNIQUEMENT en JSON, sans aucun texte avant ou après :
{{"resume_partie": "résumé", "points_cles": ["point 1", "point 2"], "transcription_propre": "texte nettoyé", "conclusion_finale": "ce qu'il faut retenir selon la vidéo, si mentionné"}}"""
    else:
        prompt = f"""Regarde et écoute cette partie ({chunk_number}/{total_chunks}) d'une vidéo éducative sur l'audiovisuel.
{LANG_INSTRUCTION}
{captions_block}
Réponds UNIQUEMENT en JSON, sans aucun texte avant ou après :
{{"resume_partie": "résumé en 2-3 phrases", "points_cles": ["point 1", "point 2"], "transcription_propre": "texte nettoyé"}}"""

    data = call_gemini([uploaded_file, prompt], log_fn, stop_event, max_output_tokens=8000, media_resolution=media_resolution)
    if not data:
        return {"resume_partie": "", "points_cles": [], "transcription_propre": ""}
    return data


SUBJECTS_INSTRUCTION = """IMPORTANT sur le nombre de sujets : la plupart des vidéos traitent d'UN SEUL sujet,
auquel cas renvoie une liste "sujets" avec un seul élément. Ne crée plusieurs éléments QUE si la vidéo
couvre clairement plusieurs sujets vraiment distincts et indépendants (ex: un épisode qui enchaîne 3
techniques de montage sans rapport entre elles). Dans le doute, ne découpe PAS."""


def analyze_single_chunk_as_full_video(uploaded_file, captions_text, description, media_resolution, log_fn, stop_event):
    """Vidéo courte (une seule tranche) : demande directement les sujets + la transcription."""
    captions_block = _captions_block(captions_text)
    desc_block = f"\nDESCRIPTION DE LA PUBLICATION (contexte, pas toujours fiable) : {description[:500]}\n" if description else ""

    prompt = f"""Regarde et écoute cette vidéo éducative sur l'audiovisuel.
{LANG_INSTRUCTION}
{captions_block}{desc_block}
{SUBJECTS_INSTRUCTION}

Pour chaque sujet : résumé (3-4 phrases), titre (max 60 car.), point clé percutant à retenir.
Donne aussi la transcription complète nettoyée (une seule fois, pour toute la vidéo).

Réponds UNIQUEMENT en JSON, sans aucun texte avant ou après :
{{"sujets": [{{"resume": "...", "titre": "...", "a_retenir": "..."}}], "transcription_propre": "..."}}"""

    data = call_gemini([uploaded_file, prompt], log_fn, stop_event, max_output_tokens=8000, media_resolution=media_resolution)
    if not data or not data.get("sujets"):
        return [{"resume": "Résumé non disponible", "titre": "Vidéo", "a_retenir": "Pas de partie à retenir."}], ""
    return data["sujets"], data.get("transcription_propre", "")


def synthesize_chunks(chunks_results, video_title, description, log_fn, stop_event):
    all_summaries = "\n".join([f"Partie {i + 1}: {c.get('resume_partie', 'N/A')}" for i, c in enumerate(chunks_results)])
    all_points = []
    for c in chunks_results:
        all_points.extend(c.get('points_cles', []))
    conclusion_finale = chunks_results[-1].get('conclusion_finale', '') if chunks_results else ''
    conclusion_block = f"\nCONCLUSION DE LA VIDÉO (à prioriser) :\n{conclusion_finale}\n" if conclusion_finale else ""
    desc_block = f"\nDESCRIPTION DE LA PUBLICATION : {description[:500]}\n" if description else ""

    prompt = f"""Tu synthétises une vidéo éducative audiovisuelle complète à partir des résumés de ses différentes parties.
{LANG_INSTRUCTION}
TITRE ORIGINAL : {video_title or "Non disponible"}
{desc_block}
RÉSUMÉS PAR PARTIE :
{all_summaries}
{conclusion_block}
POINTS CLÉS :
{chr(10).join([f"- {p}" for p in all_points[:10]])}

{SUBJECTS_INSTRUCTION}
Pour chaque sujet : résumé global (3-4 phrases), titre (max 60 car.), point clé principal
(utilise la conclusion de la vidéo si elle est pertinente pour ce sujet).

Réponds UNIQUEMENT en JSON, sans aucun texte avant ou après :
{{"sujets": [{{"resume": "...", "titre": "...", "a_retenir": "..."}}]}}"""

    data = call_gemini(prompt, log_fn, stop_event, max_output_tokens=1500)
    if not data or not data.get("sujets"):
        fallback = " ".join([c.get('resume_partie', '')[:100] for c in chunks_results[:3] if c.get('resume_partie')])
        return [{
            "resume": fallback or "Analyse non disponible",
            "titre": video_title or "Vidéo",
            "a_retenir": conclusion_finale or (all_points[0] if all_points else "Contenu non analysé")
        }]
    return data["sujets"]


def analyze_full_video_cached(work_id, video_path, video_title, description, captions_text, chunk_duration,
                              media_resolution, log_fn, stop_event):
    """Analyse résumable, multi-sujets : chaque tranche déjà analysée est réutilisée depuis le
    cache disque. Renvoie (liste_de_sujets, transcription_complete)."""
    chunk_paths, duration = split_video_cached(work_id, video_path, chunk_duration, log_fn)
    total_chunks = len(chunk_paths)

    if duration:
        log_fn(f"⏱️ Durée : {int(duration // 60)} min {int(duration % 60)} s — {total_chunks} tranche(s)"
               f" — résolution {media_resolution}")

    state = load_work_state(work_id)
    state.setdefault("chunks", {})

    if total_chunks == 1:
        key = "1"
        if key in state["chunks"] and state["chunks"][key]:
            log_fn("   ♻️ Vidéo déjà analysée (reprise depuis le cache).")
            cached = state["chunks"][key]
            return cached["sujets"], cached["transcription_propre"]

        uploaded = upload_and_wait(chunk_paths[0], log_fn, stop_event)
        if not uploaded:
            sujets, transcription = [{"resume": "Résumé non disponible", "titre": video_title or "Vidéo", "a_retenir": "Pas de partie à retenir."}], ""
        else:
            try:
                sujets, transcription = analyze_single_chunk_as_full_video(uploaded, captions_text, description, media_resolution, log_fn, stop_event)
            finally:
                try:
                    gemini_client.files.delete(name=uploaded.name)
                except Exception:
                    pass

        state["chunks"][key] = {"sujets": sujets, "transcription_propre": transcription}
        save_work_state(work_id, state)
        return sujets, transcription

    # Vidéo longue : chaque tranche analysée séparément, puis synthèse multi-sujets
    for i, chunk_path in enumerate(chunk_paths, 1):
        key = str(i)
        if key in state["chunks"] and state["chunks"][key]:
            log_fn(f"   ♻️ Partie {i}/{total_chunks} déjà analysée (reprise depuis le cache).")
            continue

        log_fn(f"   🔄 Partie {i}/{total_chunks}...")
        uploaded = upload_and_wait(chunk_path, log_fn, stop_event)
        if not uploaded:
            result = {"resume_partie": "", "points_cles": [], "transcription_propre": ""}
        else:
            try:
                result = analyze_video_chunk(uploaded, i, total_chunks, captions_text, media_resolution, log_fn, stop_event)
            finally:
                try:
                    gemini_client.files.delete(name=uploaded.name)
                except Exception:
                    pass
        state["chunks"][key] = result
        save_work_state(work_id, state)

    chunks_results = [state["chunks"][str(i)] for i in range(1, total_chunks + 1)]
    transcription_propre = "\n\n".join([c.get('transcription_propre', '') for c in chunks_results if c.get('transcription_propre')])

    if state.get("synthesis"):
        log_fn("   ♻️ Synthèse déjà faite (reprise depuis le cache).")
        sujets = state["synthesis"]
    else:
        log_fn("   🔄 Synthèse finale...")
        sujets = synthesize_chunks(chunks_results, video_title, description, log_fn, stop_event)
        state["synthesis"] = sujets
        save_work_state(work_id, state)

    return sujets, transcription_propre


def estimate_chunk_count(duration_seconds, chunk_duration):
    if not duration_seconds:
        return 1
    return int(duration_seconds // chunk_duration) + (1 if duration_seconds % chunk_duration > 0 else 0)


def choose_category_with_gemini(summary, a_retenir, existing_categories, log_fn, stop_event):
    categories_list = "\n".join(f"- {c}" for c in existing_categories) or "(aucune catégorie existante pour l'instant)"
    prompt = f"""Tu classes une vidéo éducative sur l'audiovisuel dans une catégorie.
{LANG_INSTRUCTION}

RÉSUMÉ : {summary}
À RETENIR : {a_retenir}

CATÉGORIES EXISTANTES :
{categories_list}

TÂCHE :
1. Choisis la catégorie existante la plus pertinente si l'une d'elles convient vraiment.
2. Si aucune ne correspond, propose une NOUVELLE catégorie courte (1-3 mots). Dernier recours seulement.

Réponds UNIQUEMENT en JSON, sans aucun texte avant ou après :
{{"categorie": "nom de la catégorie", "nouvelle": true ou false}}"""

    data = call_gemini(prompt, log_fn, stop_event, max_output_tokens=200, temperature=0.2)
    if not data or not data.get("categorie"):
        log_fn("⚠️ Choix de catégorie impossible.")
        return (existing_categories[0] if existing_categories else DEFAULT_CATEGORY), False

    categorie = data["categorie"].strip()
    est_nouvelle = bool(data.get("nouvelle")) and categorie not in existing_categories
    return categorie, est_nouvelle


def compare_with_gemini(new_summary, existing_pages_summaries, log_fn, stop_event):
    existing_list = "\n".join(
        [f"- Page ID: {p['id']}, Titre: {p['title']}, Résumé: {p['summary']}" for p in existing_pages_summaries])
    prompt = f"""Tu es un expert en analyse de contenu vidéo.
{LANG_INSTRUCTION}

NOUVEAU CONTENU :
{new_summary}

PAGES EXISTANTES :
{existing_list}

TÂCHE : détermine si le nouveau contenu parle du MÊME SUJET qu'une page existante (même si elle
est formulée différemment, ou si c'est un approfondissement/une suite du même sujet).

Réponds UNIQUEMENT en JSON, sans aucun texte avant ou après :
{{"action": "UPDATE" ou "CREATE_NEW", "page_id": "l'ID de la page ou null", "raison": "explication courte"}}"""

    data = call_gemini(prompt, log_fn, stop_event, max_output_tokens=300, temperature=0.2)
    if not data:
        return {"action": "CREATE_NEW", "page_id": None, "raison": "Erreur d'analyse"}
    return data


def synthesize_sources_with_gemini(existing_content, new_summary, new_transcription, new_a_retenir, log_fn, stop_event):
    """Fusionne plusieurs sources en une vraie synthèse (pas juste la concaténation de résumés) :
    recoupe les informations, souligne points communs/complémentaires/contradictions éventuelles."""
    prompt = f"""Tu es un expert qui construit une base de connaissances sur l'audiovisuel à partir de
PLUSIEURS vidéos traitant du même sujet.
{LANG_INSTRUCTION}

SYNTHÈSE EXISTANTE (basée sur une ou plusieurs sources précédentes) :
{existing_content}

NOUVELLE SOURCE À INTÉGRER :
Résumé : {new_summary}
À retenir : {new_a_retenir}
Extrait de transcription : {new_transcription[:500]}...

TÂCHE : ne te contente PAS de résumer chaque source séparément. Construis une synthèse UNIFIÉE :
- Fusionne les informations qui se recoupent (ne les répète pas deux fois)
- Mets en avant les angles complémentaires entre les sources
- Signale explicitement si les sources se contredisent sur un point
- Le résultat doit se lire comme un article de connaissance cohérent, pas comme une liste de résumés

Réponds UNIQUEMENT en JSON, sans aucun texte avant ou après :
{{"synthese": "...", "a_retenir": "...", "transcription_fusionnee": "..."}}"""

    data = call_gemini(prompt, log_fn, stop_event, max_output_tokens=2500)
    if not data:
        return new_summary, new_a_retenir, new_transcription
    return (
        data.get("synthese", new_summary),
        data.get("a_retenir", new_a_retenir),
        data.get("transcription_fusionnee", new_transcription)
    )


# ============================================================
# TRAITEMENT D'UN SUJET (commun aux deux onglets)
# ============================================================

def process_one_subject(video_url, subject, transcription_propre, data_source_id, log_fn, stop_event):
    """Prend un sujet détecté (résumé/titre/à retenir) et le fait atterrir dans Notion :
    fusion avec une page existante sur le même sujet, ou création d'une nouvelle page."""
    summary = subject.get("resume", "Résumé non disponible")
    title = subject.get("titre", "Vidéo")
    a_retenir = subject.get("a_retenir", "Pas de partie à retenir.")

    log_fn(f"\n📝 Sujet : {title}")
    log_fn(f"   Résumé : {summary[:150]}...")

    existing_categories = get_existing_categories(data_source_id, log_fn)
    video_type, is_new_category = choose_category_with_gemini(summary, a_retenir, existing_categories, log_fn, stop_event)
    log_fn(("   ✨ Nouvelle catégorie créée : « " if is_new_category else "   📂 Catégorie : « ") + video_type + " »")

    existing_pages = search_existing_pages(data_source_id, video_type, log_fn)
    if existing_pages:
        pages_with_summaries = []
        for page in existing_pages[:MAX_PAGES_FOR_MATCHING]:
            content = get_page_content(page["id"], log_fn)
            if content:
                pages_with_summaries.append({"id": page["id"], "title": page["title"], "summary": content[:500]})

        if pages_with_summaries:
            comparison = compare_with_gemini(summary, pages_with_summaries, log_fn, stop_event)
            if comparison.get("action") == "UPDATE" and comparison.get("page_id"):
                page_id = comparison["page_id"]
                log_fn(f"   ✨ Sujet déjà couvert par une page existante : {comparison.get('raison')}")

                existing_content = get_page_content(page_id, log_fn)
                existing_sources = get_page_source_urls(page_id, log_fn)
                all_sources = existing_sources + ([video_url] if video_url not in existing_sources else [])

                synthese, merged_a_retenir, merged_transcription = synthesize_sources_with_gemini(
                    existing_content, summary, transcription_propre, a_retenir, log_fn, stop_event
                )

                update_page_properties(page_id, title, video_type, all_sources, log_fn)
                if replace_page_content(page_id, synthese, merged_a_retenir, merged_transcription, all_sources, log_fn, updated=True):
                    log_fn(f"   ✅ Page mise à jour avec {len(all_sources)} source(s) au total.")
                    return "updated", page_id
                return "error", None
            else:
                log_fn(f"   📄 Nouveau sujet (pas de correspondance) : {comparison.get('raison')}")

    send_result = send_to_notion(video_url, transcription_propre, summary, title, video_type, data_source_id, a_retenir, log_fn)
    if send_result:
        log_fn("   ✅ Nouvelle page créée.")
        return "created", send_result.get("id")
    return "error", None


def apply_primary_subject_to_page(page_id, video_url, subject, transcription_propre, data_source_id, log_fn, stop_event):
    """Met à jour une page EXISTANTE en place (identité préservée), au lieu de chercher une
    correspondance ailleurs. Utilisé pour le sujet principal lors d'une migration/re-analyse :
    la page qu'on migre doit rester la même page, jamais remplacée par une nouvelle ni fusionnée
    ailleurs. Sert aussi pour les pages 'placeholder' créées par la surveillance YouTube pour les
    vidéos trop longues : elles sont indiscernables d'une page à migrer, donc ramassées pareil."""
    summary = subject.get("resume", "Résumé non disponible")
    title = subject.get("titre", "Vidéo")
    a_retenir = subject.get("a_retenir", "Pas de partie à retenir.")

    existing_categories = get_existing_categories(data_source_id, log_fn)
    video_type, is_new_category = choose_category_with_gemini(summary, a_retenir, existing_categories, log_fn, stop_event)
    log_fn(("   ✨ Nouvelle catégorie créée : « " if is_new_category else "   📂 Catégorie : « ") + video_type + " »")

    existing_sources = get_page_source_urls(page_id, log_fn)
    all_sources = existing_sources if video_url in existing_sources else existing_sources + [video_url]

    update_page_properties(page_id, title, video_type, all_sources, log_fn)
    return replace_page_content(page_id, summary, a_retenir, transcription_propre, all_sources, log_fn, updated=True)


# ============================================================
# CHAÎNES YOUTUBE (lecture base "Chaînes YouTube" + listing historique)
# ============================================================

def list_watched_channels(channels_ds_id, log_fn):
    pages = []
    payload = {"page_size": 100}
    while True:
        response = notion_request("POST", f"https://api.notion.com/v1/data_sources/{channels_ds_id}/query",
                                  log_fn, json=payload)
        response.raise_for_status()
        data = response.json()
        pages.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        payload["start_cursor"] = data["next_cursor"]

    channels = []
    for page in pages:
        props = page.get("properties", {})
        title_list = props.get("Nom", {}).get("title", [])
        name = title_list[0]["text"]["content"] if title_list else "Sans nom"
        channel_url = props.get("Chaîne", {}).get("url")
        active = bool(props.get("Actif", {}).get("checkbox"))
        channels.append({"page_id": page["id"], "name": name, "channel_url": channel_url, "active": active})
    return channels


def get_channel_all_videos(channel_url, log_fn):
    """Liste TOUTES les vidéos d'une chaîne (titre + id, pas de description) via yt-dlp
    en mode liste à plat : rapide, gratuit, zéro quota Gemini. Peut renvoyer des centaines
    d'entrées pour une grosse chaîne."""
    target = (channel_url or "").rstrip("/")
    if target and not target.endswith("/videos"):
        target += "/videos"
    try:
        result = subprocess.run(["yt-dlp", "--flat-playlist", "-J", target],
                                capture_output=True, text=True, check=True, timeout=180)
        data = json.loads(result.stdout)
        entries = data.get("entries") or []
        videos = []
        for e in entries:
            vid = e.get("id")
            if not vid:
                continue
            videos.append({
                "video_id": vid,
                "title": e.get("title") or "Sans titre",
                "url": f"https://www.youtube.com/watch?v={vid}",
            })
        return videos
    except Exception as e:
        log_fn(f"⚠️ Impossible de lister les vidéos de la chaîne : {e}")
        return []


def triage_video_worth_it(title, description, log_fn, stop_event):
    """Même logique de tri que la surveillance quotidienne : contenu pertinent pour le
    projet, ET uniquement DaVinci Resolve (les vidéos sur un autre logiciel de montage
    précis sont écartées avant tout traitement coûteux)."""
    prompt = f"""Tu aides à filtrer automatiquement des vidéos YouTube pour un projet personnel
d'archivage de connaissances sur l'audiovisuel (montage, étalonnage, son, effets, storytelling, etc.).
{LANG_INSTRUCTION}

TITRE : {title}
DESCRIPTION : {description[:500] if description else "(non disponible)"}

TÂCHE 1 (contenu) : cette vidéo a-t-elle un contenu éducatif/technique qui mérite d'être archivé et
analysé en détail (tutoriel, explication technique, retour d'expérience, comparatif...) ?
Réponds NON si c'est une simple annonce, un vlog sans contenu technique, une vidéo communautaire,
un live replay sans substance, ou du contenu publicitaire/sponsorisé pur.

TÂCHE 2 (logiciel) : seul DaVinci Resolve est utilisé par la personne qui archive. Si le titre ou
la description mentionne clairement un AUTRE logiciel de montage précis (Premiere Pro, Final Cut
Pro, Avid Media Composer, CapCut, Vegas Pro, etc.) sans mentionner Resolve, cette vidéo n'est PAS
pertinente. Si aucun logiciel précis n'est mentionné, ou si Resolve est mentionné, c'est pertinent.

Réponds UNIQUEMENT en JSON, sans aucun texte avant ou après :
{{"vaut_le_coup": true ou false, "raison": "explication très courte", "autre_logiciel_detecte": "nom du logiciel si un AUTRE logiciel précis a été détecté, sinon chaîne vide"}}"""

    data = call_gemini(prompt, log_fn, stop_event, max_output_tokens=200, temperature=0.1)
    if not data:
        return True, "Tri impossible (réponse invalide), traitée par défaut par précaution"

    autre_logiciel = (data.get("autre_logiciel_detecte") or "").strip()
    if autre_logiciel:
        return False, f"Logiciel non pertinent détecté : {autre_logiciel} (seul DaVinci Resolve est suivi)"
    return bool(data.get("vaut_le_coup", True)), data.get("raison", "")


def load_backfill_state():
    if os.path.isfile(BACKFILL_STATE_FILE):
        try:
            with open(BACKFILL_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_backfill_state(state):
    with open(BACKFILL_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================================================
# VERROU CROISÉ ENTRE ONGLETS
# ============================================================

class AppLock:
    def __init__(self):
        self.busy_tab = None

    def acquire(self, tab_name):
        if self.busy_tab is not None and self.busy_tab != tab_name:
            return False
        self.busy_tab = tab_name
        return True

    def release(self, tab_name):
        if self.busy_tab == tab_name:
            self.busy_tab = None


# ============================================================
# ONGLET 1 : AJOUTER UNE VIDÉO
# ============================================================

class AddVideoTab(ctk.CTkFrame):
    def __init__(self, master, app_lock):
        super().__init__(master, fg_color="transparent")
        self.app_lock = app_lock
        self.stop_event = threading.Event()

        ctk.CTkLabel(self, text="URL de la vidéo ou chemin local :", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=20, pady=(15, 5))
        self.url_entry = ctk.CTkEntry(self, width=660, height=40, placeholder_text="https://youtube.com/watch?v=... ou un lien Instagram",
                                      font=ctk.CTkFont(size=13))
        self.url_entry.pack(padx=20, pady=(0, 10))

        ctk.CTkLabel(self, text="La catégorie est choisie automatiquement par l'IA. Si la vidéo couvre plusieurs "
                                "sujets distincts, plusieurs pages peuvent être créées.",
                     font=ctk.CTkFont(size=11), text_color="gray", wraplength=660, justify="left").pack(anchor="w", padx=20, pady=(0, 15))

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(0, 15))
        self.start_btn = ctk.CTkButton(btn_frame, text="🚀 Lancer le traitement", command=self.start,
                                       height=42, font=ctk.CTkFont(size=14, weight="bold"),
                                       fg_color="#2563eb", hover_color="#1d4ed8")
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.stop_btn = ctk.CTkButton(btn_frame, text="⏸️ Arrêter", command=self.request_stop,
                                      height=42, font=ctk.CTkFont(size=14, weight="bold"),
                                      fg_color="#78716c", hover_color="#57534e", state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=5)

        self.progress_bar = ctk.CTkProgressBar(self, width=660, height=20, corner_radius=10)
        self.progress_bar.pack(padx=20, pady=(0, 8))
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(self, text="En attente...", font=ctk.CTkFont(size=12), text_color="gray")
        self.progress_label.pack()

        ctk.CTkLabel(self, text="Logs :", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=20, pady=(10, 5))
        self.log_text = ctk.CTkTextbox(self, width=660, height=220, font=ctk.CTkFont(size=12, family="Consolas"), corner_radius=10)
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        self.log_text.configure(state="disabled")

        self.log = make_logger(self.log_text, self)

    def update_progress(self, value, label):
        self.progress_bar.set(max(0, min(value, 100)) / 100)
        self.progress_label.configure(text=label)
        if "Erreur" in label or "pause" in label.lower() or "surchargé" in label:
            self.progress_label.configure(text_color="#ef4444")
        elif "✅" in label:
            self.progress_label.configure(text_color="#22c55e")
        else:
            self.progress_label.configure(text_color="gray")
        self.update()

    def start(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Erreur", "Merci de renseigner une URL ou un chemin de vidéo.")
            return
        if not self.app_lock.acquire("add"):
            messagebox.showwarning("Occupé", "Une migration est en cours dans l'autre onglet.")
            return
        if not check_dependencies(self.log):
            self.app_lock.release("add")
            messagebox.showerror("Erreur", "ffmpeg, ffprobe ou yt-dlp introuvable dans le PATH.")
            return

        self.stop_event.clear()
        self.start_btn.configure(state="disabled", text="⏳ Traitement en cours...")
        self.stop_btn.configure(state="normal")
        threading.Thread(target=self.run_workflow, args=(url,), daemon=True).start()

    def request_stop(self):
        self.stop_event.set()
        self.stop_btn.configure(state="disabled", text="⏳ Arrêt en cours...")

    def run_workflow(self, video_url):
        work_id = get_work_id_for_url(video_url)
        try:
            self.update_progress(0, "Initialisation...")
            self.log("🔍 Récupération du data_source_id...")
            data_source_id = get_data_source_id(DATABASE_ID, self.log)
            self.update_progress(5, "Configuration OK")

            self.log("🔎 Vérification des doublons...")
            existing = find_page_by_source_url(data_source_id, video_url, self.log)
            if existing:
                self.log(f"⚠️ Cette URL est déjà une source de la page « {existing['title']} ». Traitement annulé (aucun quota utilisé).")
                self.update_progress(0, "Déjà existant")
                return

            self.update_progress(10, "Métadonnées...")
            metadata = get_video_metadata(video_url, self.log)
            self.log("📡 Recherche de sous-titres officiels...")
            captions_text = try_fetch_captions(work_id, video_url, self.log)
            if captions_text:
                self.log("   ✅ Sous-titres trouvés : résolution vidéo réduite, tranches plus longues (moins de quota).")
                chunk_duration = CHUNK_DURATION_WITH_CAPTIONS
                media_resolution = MEDIA_RESOLUTION_WITH_CAPTIONS
            else:
                self.log("   ℹ️ Pas de sous-titres officiels : analyse 100% vidéo (résolution haute).")
                chunk_duration = CHUNK_DURATION_NO_CAPTIONS
                media_resolution = MEDIA_RESOLUTION_NO_CAPTIONS

            self.update_progress(15, "Téléchargement de la vidéo...")
            video_path, video_title = download_video_cached(work_id, video_url, metadata.get("title") or "Vidéo", self.log)
            if not video_path:
                self.update_progress(0, "Erreur")
                return
            self.log(f"✅ Titre : {video_title}")
            self.update_progress(30, "Vidéo téléchargée")

            duration = get_video_duration(video_path)
            nb_chunks = estimate_chunk_count(duration, chunk_duration)
            if nb_chunks > 1:
                proceed = messagebox.askyesno(
                    "Vidéo longue",
                    f"Cette vidéo dure environ {int(duration // 60)} min et sera découpée en {nb_chunks} parties "
                    f"(~{nb_chunks + 1} appels à Gemini).\n\nContinuer ?"
                )
                if not proceed:
                    self.log("⏸️ Traitement annulé par l'utilisateur.")
                    self.update_progress(0, "Annulé")
                    return

            self.update_progress(35, "Analyse par Gemini...")
            self.log("🤖 Analyse de la vidéo avec Gemini...")
            sujets, transcription_propre = analyze_full_video_cached(
                work_id, video_path, video_title, metadata.get("description", ""), captions_text,
                chunk_duration, media_resolution, self.log, self.stop_event
            )

            if len(sujets) > 1:
                self.log(f"\n🧩 {len(sujets)} sujets distincts détectés : {len(sujets)} pages vont être créées/mises à jour.")

            for i, subject in enumerate(sujets, 1):
                self.update_progress(70 + int(25 * i / len(sujets)), f"Sujet {i}/{len(sujets)}...")
                process_one_subject(video_url, subject, transcription_propre, data_source_id, self.log, self.stop_event)

            clear_work(work_id, self.log)
            self.update_progress(100, "✅ Terminé avec succès !")

        except QuotaExceededError as e:
            self.log(f"\n⏳ Quota Gemini atteint : {e}")
            self.log("💡 Le travail déjà effectué est sauvegardé. Relance plus tard avec LA MÊME URL pour reprendre.")
            self.update_progress(0, "⏳ Quota atteint - en pause")
        except ServiceOverloadedError as e:
            self.log(f"\n🚦 Gemini reste surchargé : {e}")
            self.log("💡 Relance dans quelques minutes avec la même URL, rien n'est perdu.")
            self.update_progress(0, "🚦 Gemini surchargé - en pause")
        except StopRequested:
            self.log("\n⏸️ Arrêt demandé. Le travail déjà effectué est sauvegardé.")
            self.update_progress(0, "⏸️ En pause (arrêt manuel)")
        except Exception as e:
            self.log(f"❌ Erreur inattendue : {e}\n{traceback.format_exc()}")
            self.update_progress(0, "Erreur")
        finally:
            self.start_btn.configure(state="normal", text="🚀 Lancer le traitement")
            self.stop_btn.configure(state="disabled", text="⏸️ Arrêter")
            self.app_lock.release("add")


# ============================================================
# ONGLET 2 : MIGRATION GROQ → GEMINI
# ============================================================

class MigrationTab(ctk.CTkFrame):
    def __init__(self, master, app_lock):
        super().__init__(master, fg_color="transparent")
        self.app_lock = app_lock
        self.stop_event = threading.Event()
        self.session_start_time = None
        self.session_pages_done = 0

        ctk.CTkLabel(
            self,
            text="Re-télécharge et ré-analyse chaque vidéo pas encore marquée « Gemini » dans Notion.\n"
                 "Reprend automatiquement au niveau de la tranche en cas d'interruption.",
            font=ctk.CTkFont(size=12), text_color="gray", justify="center"
        ).pack(pady=(15, 15))

        stats_frame = ctk.CTkFrame(self)
        stats_frame.pack(fill="x", padx=20, pady=(0, 15))
        self.stat_labels = {}
        for key, label in [("total", "Total"), ("done", "Migrées"), ("failed", "Échouées"), ("remaining", "Restantes")]:
            col = ctk.CTkFrame(stats_frame, fg_color="transparent")
            col.pack(side="left", expand=True, fill="both", padx=10, pady=10)
            value_label = ctk.CTkLabel(col, text="-", font=ctk.CTkFont(size=20, weight="bold"))
            value_label.pack()
            ctk.CTkLabel(col, text=label, font=ctk.CTkFont(size=11), text_color="gray").pack()
            self.stat_labels[key] = value_label

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(0, 10))
        self.start_btn = ctk.CTkButton(btn_frame, text="🚀 Lancer / Reprendre", command=self.start,
                                       height=42, font=ctk.CTkFont(size=14, weight="bold"),
                                       fg_color="#2563eb", hover_color="#1d4ed8")
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.stop_btn = ctk.CTkButton(btn_frame, text="⏸️ Arrêter", command=self.request_stop,
                                      height=42, font=ctk.CTkFont(size=14, weight="bold"),
                                      fg_color="#78716c", hover_color="#57534e", state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=5)

        btn_frame2 = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame2.pack(fill="x", padx=20, pady=(0, 15))
        self.retry_btn = ctk.CTkButton(btn_frame2, text="🔁 Réessayer les échecs", command=self.retry_failed,
                                       height=36, font=ctk.CTkFont(size=13), fg_color="#0d9488", hover_color="#0f766e")
        self.retry_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.clear_cache_btn = ctk.CTkButton(btn_frame2, text="🧹 Vider le cache vidéo", command=self.clear_cache,
                                             height=36, font=ctk.CTkFont(size=13), fg_color="#78716c", hover_color="#57534e")
        self.clear_cache_btn.pack(side="left", expand=True, fill="x", padx=5)

        self.progress_bar = ctk.CTkProgressBar(self, width=660, height=20, corner_radius=10)
        self.progress_bar.pack(padx=20, pady=(0, 8))
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(self, text="En attente...", font=ctk.CTkFont(size=12), text_color="gray")
        self.progress_label.pack()

        ctk.CTkLabel(self, text="Logs :", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=20, pady=(10, 5))
        self.log_text = ctk.CTkTextbox(self, width=660, height=190, font=ctk.CTkFont(size=12, family="Consolas"), corner_radius=10)
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        self.log_text.configure(state="disabled")

        self.log = make_logger(self.log_text, self)
        self.refresh_stats_display()

    def update_progress(self, value, label):
        self.progress_bar.set(max(0, min(value, 100)) / 100)
        self.progress_label.configure(text=label)
        if "Erreur" in label or "atteint" in label or "surchargé" in label or "pause" in label.lower():
            self.progress_label.configure(text_color="#ef4444")
        elif "✅" in label:
            self.progress_label.configure(text_color="#22c55e")
        else:
            self.progress_label.configure(text_color="gray")
        self.update()

    def refresh_stats_display(self, total=None, done=None, failed=None, remaining=None):
        self.stat_labels["total"].configure(text=str(total) if total is not None else "-")
        self.stat_labels["done"].configure(text=str(done) if done is not None else "-")
        self.stat_labels["failed"].configure(text=str(failed) if failed is not None else "-")
        self.stat_labels["remaining"].configure(text=str(remaining) if remaining is not None else "-")
        self.update()

    def estimate_time_label(self, done_this_session, remaining):
        if done_this_session == 0 or self.session_start_time is None:
            return ""
        elapsed = (datetime.now() - self.session_start_time).total_seconds()
        eta = timedelta(seconds=int((elapsed / done_this_session) * remaining))
        return f" (≈ {str(eta)} restant)"

    def start(self):
        if not self.app_lock.acquire("migration"):
            messagebox.showwarning("Occupé", "Un ajout de vidéo est en cours dans l'autre onglet.")
            return
        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.retry_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal", text="⏸️ Arrêter")
        self.session_start_time = datetime.now()
        self.session_pages_done = 0
        threading.Thread(target=self.run_migration, daemon=True).start()

    def request_stop(self):
        self.stop_event.set()
        self.stop_btn.configure(state="disabled", text="⏳ Arrêt en cours...")

    def retry_failed(self):
        failures = load_failures()
        count = len(failures)
        if os.path.isfile(MIGRATION_FAILURES_FILE):
            os.remove(MIGRATION_FAILURES_FILE)
        self.log(f"🔁 {count} page(s) en échec repassée(s) en attente.")

    def clear_cache(self):
        if not messagebox.askyesno("Confirmation", "Vide le cache de vidéos/tranches téléchargées. Continuer ?"):
            return
        clear_all_cache(self.log)

    def run_migration(self):
        try:
            if not check_dependencies(self.log):
                return
            if not test_connections(self.log, self.stop_event):
                return

            self.log("📄 Récupération de toutes les pages existantes...")
            data_source_id = get_data_source_id(DATABASE_ID, self.log)
            pages = list_all_pages(data_source_id, self.log)
            total = len(pages)

            known_categories = get_existing_categories(data_source_id, self.log)
            failures = load_failures()

            to_process = []
            done_count = 0
            for page in pages:
                title, video_url, video_type, is_done = extract_page_info(page)
                if is_done:
                    done_count += 1
                    continue
                to_process.append((page["id"], title, video_url, video_type))

            failed_final = sum(1 for pid, *_ in to_process if failures.get(pid, {}).get("attempts", 0) >= MAX_FAILED_ATTEMPTS)
            remaining = len(to_process) - failed_final
            self.refresh_stats_display(total, done_count, failed_final, remaining)
            self.log(f"✅ {total} page(s) au total — {done_count} déjà migrée(s), {remaining} restante(s), {failed_final} abandonnée(s).")

            for index, (page_id, title, video_url, video_type) in enumerate(to_process, 1):
                if self.stop_event.is_set():
                    self.log("\n⏸️ Arrêt demandé, migration mise en pause proprement.")
                    self.update_progress((done_count / total) * 100 if total else 0, "⏸️ En pause (arrêt manuel)")
                    return

                if failures.get(page_id, {}).get("attempts", 0) >= MAX_FAILED_ATTEMPTS:
                    continue

                self.update_progress((done_count / total) * 100 if total else 0,
                                     f"Page {index}/{len(to_process)}{self.estimate_time_label(self.session_pages_done, remaining)}")
                self.log(f"\n➡️ [{index}/{len(to_process)}] {title}")

                if not video_url:
                    self.log("   ⚠️ Pas d'URL source, ignorée définitivement.")
                    record_failure(failures, page_id, title, "no_url")
                    continue

                work_id = page_id
                try:
                    metadata = get_video_metadata(video_url, self.log)
                    captions_text = try_fetch_captions(work_id, video_url, self.log)
                    if captions_text:
                        chunk_duration, media_resolution = CHUNK_DURATION_WITH_CAPTIONS, MEDIA_RESOLUTION_WITH_CAPTIONS
                    else:
                        chunk_duration, media_resolution = CHUNK_DURATION_NO_CAPTIONS, MEDIA_RESOLUTION_NO_CAPTIONS

                    self.log("   ⬇️ Téléchargement de la vidéo...")
                    video_path, video_title = download_video_cached(work_id, video_url, title, self.log)
                    if not video_path:
                        raise RuntimeError("Échec du téléchargement")

                    self.log("   🤖 Analyse avec Gemini (image + son)...")
                    sujets, transcription = analyze_full_video_cached(
                        work_id, video_path, video_title, metadata.get("description", ""), captions_text,
                        chunk_duration, media_resolution, self.log, self.stop_event
                    )

                    if len(sujets) > 1:
                        self.log(f"   🧩 {len(sujets)} sujets distincts détectés : le 1er met à jour "
                                 f"cette page, les suivants créent/rejoignent d'autres pages.")

                    primary_subject, *extra_subjects = sujets
                    ok = apply_primary_subject_to_page(page_id, video_url, primary_subject, transcription,
                                                       data_source_id, self.log, self.stop_event)
                    if not ok:
                        raise RuntimeError("Échec de la mise à jour de la page")

                    for subject in extra_subjects:
                        process_one_subject(video_url, subject, transcription, data_source_id, self.log, self.stop_event)

                    done_count += 1
                    self.session_pages_done += 1
                    clear_failure(failures, page_id)
                    clear_work(work_id)
                    self.log(f"   ✅ Page migrée.")

                except QuotaExceededError as e:
                    self.log(f"\n⏳ Quota Gemini atteint : {e}")
                    self.log("💡 Le travail déjà fait est sauvegardé (cache conservé). Relance plus tard.")
                    self.update_progress((done_count / total) * 100 if total else 0, "⏳ Quota atteint - en pause")
                    return

                except ServiceOverloadedError as e:
                    self.log(f"\n🚦 Gemini reste surchargé après {OVERLOAD_MAX_ATTEMPTS} tentatives : {e}")
                    self.update_progress((done_count / total) * 100 if total else 0, "🚦 Gemini surchargé - en pause")
                    return

                except StopRequested:
                    self.log("\n⏸️ Arrêt demandé, cache conservé pour cette page.")
                    self.update_progress((done_count / total) * 100 if total else 0, "⏸️ En pause (arrêt manuel)")
                    return

                except Exception as e:
                    attempts = record_failure(failures, page_id, title, e)
                    self.log(f"   ❌ Erreur (essai {attempts}/{MAX_FAILED_ATTEMPTS}) : {e}")
                    if attempts >= MAX_FAILED_ATTEMPTS:
                        self.log("   🧹 Abandon définitif de cette page, nettoyage du cache associé.")
                        clear_work(work_id)

                time.sleep(DELAY_BETWEEN_PAGES)

            self.log("\n🎉 Migration terminée : toutes les pages traitables ont été migrées !")
            self.update_progress(100, "✅ Terminé")

        except Exception as e:
            self.log(f"❌ Erreur inattendue : {e}\n{traceback.format_exc()}")
            self.update_progress(0, "Erreur")
        finally:
            self.start_btn.configure(state="normal", text="🚀 Lancer / Reprendre")
            self.retry_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled", text="⏸️ Arrêter")
            self.app_lock.release("migration")


# ============================================================
# ONGLET 3 : HISTORIQUE DES CHAÎNES YOUTUBE
# ============================================================

class HistoryTab(ctk.CTkFrame):
    def __init__(self, master, app_lock):
        super().__init__(master, fg_color="transparent")
        self.app_lock = app_lock
        self.stop_event = threading.Event()
        self.session_start_time = None
        self.session_done = 0

        ctk.CTkLabel(
            self,
            text="Récupère tout l'historique des chaînes YouTube actives (base \"Chaînes YouTube\").\n"
                 "Chaque vidéo passe d'abord par un tri IA léger (pertinence + DaVinci Resolve "
                 "uniquement) avant le traitement complet, pour ne pas gaspiller de quota.",
            font=ctk.CTkFont(size=12), text_color="gray", justify="center", wraplength=660
        ).pack(pady=(15, 15))

        stats_frame = ctk.CTkFrame(self)
        stats_frame.pack(fill="x", padx=20, pady=(0, 15))
        self.stat_labels = {}
        for key, label in [("total", "Total"), ("added", "Ajoutées"), ("skipped", "Ignorées"), ("remaining", "Restantes")]:
            col = ctk.CTkFrame(stats_frame, fg_color="transparent")
            col.pack(side="left", expand=True, fill="both", padx=10, pady=10)
            value_label = ctk.CTkLabel(col, text="-", font=ctk.CTkFont(size=20, weight="bold"))
            value_label.pack()
            ctk.CTkLabel(col, text=label, font=ctk.CTkFont(size=11), text_color="gray").pack()
            self.stat_labels[key] = value_label

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(0, 10))
        self.start_btn = ctk.CTkButton(btn_frame, text="🚀 Lancer / Reprendre", command=self.start,
                                       height=42, font=ctk.CTkFont(size=14, weight="bold"),
                                       fg_color="#2563eb", hover_color="#1d4ed8")
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.stop_btn = ctk.CTkButton(btn_frame, text="⏸️ Arrêter", command=self.request_stop,
                                      height=42, font=ctk.CTkFont(size=14, weight="bold"),
                                      fg_color="#78716c", hover_color="#57534e", state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=5)

        btn_frame2 = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame2.pack(fill="x", padx=20, pady=(0, 15))
        self.retry_btn = ctk.CTkButton(btn_frame2, text="🔁 Réessayer les échecs", command=self.retry_failed,
                                       height=36, font=ctk.CTkFont(size=13), fg_color="#0d9488", hover_color="#0f766e")
        self.retry_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.reset_btn = ctk.CTkButton(btn_frame2, text="🗑️ Réinitialiser l'historique", command=self.reset_state,
                                       height=36, font=ctk.CTkFont(size=13), fg_color="#b91c1c", hover_color="#991b1b")
        self.reset_btn.pack(side="left", expand=True, fill="x", padx=5)

        self.progress_bar = ctk.CTkProgressBar(self, width=660, height=20, corner_radius=10)
        self.progress_bar.pack(padx=20, pady=(0, 8))
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(self, text="En attente...", font=ctk.CTkFont(size=12), text_color="gray")
        self.progress_label.pack()

        ctk.CTkLabel(self, text="Logs :", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=20, pady=(10, 5))
        self.log_text = ctk.CTkTextbox(self, width=660, height=190, font=ctk.CTkFont(size=12, family="Consolas"), corner_radius=10)
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        self.log_text.configure(state="disabled")

        self.log = make_logger(self.log_text, self)
        self.refresh_stats_display()

    def update_progress(self, value, label):
        self.progress_bar.set(max(0, min(value, 100)) / 100)
        self.progress_label.configure(text=label)
        if "Erreur" in label or "atteint" in label or "surchargé" in label or "pause" in label.lower():
            self.progress_label.configure(text_color="#ef4444")
        elif "✅" in label:
            self.progress_label.configure(text_color="#22c55e")
        else:
            self.progress_label.configure(text_color="gray")
        self.update()

    def refresh_stats_display(self, total=None, added=None, skipped=None, remaining=None):
        self.stat_labels["total"].configure(text=str(total) if total is not None else "-")
        self.stat_labels["added"].configure(text=str(added) if added is not None else "-")
        self.stat_labels["skipped"].configure(text=str(skipped) if skipped is not None else "-")
        self.stat_labels["remaining"].configure(text=str(remaining) if remaining is not None else "-")
        self.update()

    def estimate_time_label(self, done_this_session, remaining):
        if done_this_session == 0 or self.session_start_time is None:
            return ""
        elapsed = (datetime.now() - self.session_start_time).total_seconds()
        eta = timedelta(seconds=int((elapsed / done_this_session) * remaining))
        return f" (≈ {str(eta)} restant)"

    def start(self):
        if not CHANNELS_DATABASE_ID:
            messagebox.showerror("Erreur", "CHANNELS_DATABASE_ID n'est pas défini dans le .env.")
            return
        if not self.app_lock.acquire("history"):
            messagebox.showwarning("Occupé", "Un autre traitement est en cours dans un autre onglet.")
            return
        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.retry_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal", text="⏸️ Arrêter")
        self.session_start_time = datetime.now()
        self.session_done = 0
        threading.Thread(target=self.run_backfill, daemon=True).start()

    def request_stop(self):
        self.stop_event.set()
        self.stop_btn.configure(state="disabled", text="⏳ Arrêt en cours...")

    def retry_failed(self):
        state = load_backfill_state()
        reset_count = 0
        for ch_state in state.values():
            reset_count += len(ch_state.get("attempts", {}))
            ch_state["attempts"] = {}
        save_backfill_state(state)
        self.log(f"🔁 {reset_count} vidéo(s) en échec repassée(s) en attente.")

    def reset_state(self):
        if not messagebox.askyesno("Confirmation",
                                   "Tout l'historique de progression sera effacé : toutes les vidéos des "
                                   "chaînes actives seront re-triées depuis le début. Continuer ?"):
            return
        if os.path.isfile(BACKFILL_STATE_FILE):
            os.remove(BACKFILL_STATE_FILE)
        self.log("🗑️ Historique réinitialisé.")
        self.refresh_stats_display()

    def run_backfill(self):
        try:
            if not check_dependencies(self.log):
                return
            if not test_connections(self.log, self.stop_event):
                return

            data_source_id = get_data_source_id(DATABASE_ID, self.log)
            channels_ds_id = get_data_source_id(CHANNELS_DATABASE_ID, self.log)
            channels = [c for c in list_watched_channels(channels_ds_id, self.log) if c["active"] and c["channel_url"]]
            self.log(f"📺 {len(channels)} chaîne(s) active(s).")

            state = load_backfill_state()
            all_items = []
            added_so_far = 0
            skipped_so_far = 0

            for channel in channels:
                self.log(f"\n🔎 Listing des vidéos : {channel['name']}...")
                videos = get_channel_all_videos(channel["channel_url"], self.log)
                self.log(f"   {len(videos)} vidéo(s) trouvée(s) sur la chaîne.")

                ch_state = state.setdefault(channel["page_id"], {"done": {}, "attempts": {}})
                for status in ch_state["done"].values():
                    if status == "added":
                        added_so_far += 1
                    elif status in ("skipped", "already_exists"):
                        skipped_so_far += 1

                for v in videos:
                    if v["video_id"] in ch_state["done"]:
                        continue
                    if ch_state["attempts"].get(v["video_id"], 0) >= MAX_FAILED_ATTEMPTS:
                        continue
                    all_items.append((channel, v, ch_state))

            save_backfill_state(state)
            total = len(all_items) + added_so_far + skipped_so_far
            self.refresh_stats_display(total, added_so_far, skipped_so_far, len(all_items))
            self.log(f"\n✅ {len(all_items)} vidéo(s) restante(s) à examiner sur {total} au total.")

            for index, (channel, video, ch_state) in enumerate(all_items, 1):
                if self.stop_event.is_set():
                    self.log("\n⏸️ Arrêt demandé, traitement mis en pause proprement.")
                    self.update_progress((added_so_far + skipped_so_far) / total * 100 if total else 0, "⏸️ En pause (arrêt manuel)")
                    return

                eta = self.estimate_time_label(self.session_done, len(all_items) - index + 1)
                self.update_progress((added_so_far + skipped_so_far) / total * 100 if total else 0,
                                     f"Vidéo {index}/{len(all_items)}{eta}")
                self.log(f"\n➡️ [{index}/{len(all_items)}] ({channel['name']}) {video['title']}")

                existing = find_page_by_source_url(data_source_id, video["url"], self.log)
                if existing:
                    self.log(f"   ℹ️ Déjà présente dans Notion : « {existing['title']} »")
                    ch_state["done"][video["video_id"]] = "already_exists"
                    save_backfill_state(state)
                    skipped_so_far += 1
                    continue

                work_id = get_work_id_for_url(video["url"])
                try:
                    worth_it, reason = triage_video_worth_it(video["title"], "", self.log, self.stop_event)
                    if not worth_it:
                        self.log(f"   ⏭️ Ignorée : {reason}")
                        ch_state["done"][video["video_id"]] = "skipped"
                        save_backfill_state(state)
                        skipped_so_far += 1
                        self.session_done += 1
                        continue

                    self.log(f"   ✅ Retenue : {reason}")
                    metadata = get_video_metadata(video["url"], self.log)
                    captions_text = try_fetch_captions(work_id, video["url"], self.log)
                    if captions_text:
                        chunk_duration, media_resolution = CHUNK_DURATION_WITH_CAPTIONS, MEDIA_RESOLUTION_WITH_CAPTIONS
                    else:
                        chunk_duration, media_resolution = CHUNK_DURATION_NO_CAPTIONS, MEDIA_RESOLUTION_NO_CAPTIONS

                    video_path, video_title = download_video_cached(work_id, video["url"], video["title"], self.log)
                    if not video_path:
                        raise RuntimeError("Échec du téléchargement")

                    sujets, transcription = analyze_full_video_cached(
                        work_id, video_path, video_title, metadata.get("description", ""), captions_text,
                        chunk_duration, media_resolution, self.log, self.stop_event
                    )
                    for subject in sujets:
                        process_one_subject(video["url"], subject, transcription, data_source_id, self.log, self.stop_event)

                    clear_work(work_id)
                    ch_state["done"][video["video_id"]] = "added"
                    save_backfill_state(state)
                    added_so_far += 1
                    self.session_done += 1
                    self.log("   ✅ Ajoutée à Notion.")

                except QuotaExceededError as e:
                    self.log(f"\n⏳ Quota Gemini atteint : {e}")
                    self.log("💡 Le travail déjà fait est sauvegardé (cache conservé). Relance plus tard.")
                    save_backfill_state(state)
                    self.update_progress((added_so_far + skipped_so_far) / total * 100 if total else 0, "⏳ Quota atteint - en pause")
                    return

                except ServiceOverloadedError as e:
                    self.log(f"\n🚦 Gemini reste surchargé après {OVERLOAD_MAX_ATTEMPTS} tentatives : {e}")
                    save_backfill_state(state)
                    self.update_progress((added_so_far + skipped_so_far) / total * 100 if total else 0, "🚦 Gemini surchargé - en pause")
                    return

                except StopRequested:
                    self.log("\n⏸️ Arrêt demandé, cache conservé pour cette vidéo.")
                    save_backfill_state(state)
                    self.update_progress((added_so_far + skipped_so_far) / total * 100 if total else 0, "⏸️ En pause (arrêt manuel)")
                    return

                except Exception as e:
                    attempts = ch_state["attempts"].get(video["video_id"], 0) + 1
                    ch_state["attempts"][video["video_id"]] = attempts
                    save_backfill_state(state)
                    self.log(f"   ❌ Erreur (essai {attempts}/{MAX_FAILED_ATTEMPTS}) : {e}")

                time.sleep(DELAY_BETWEEN_PAGES)

            self.refresh_stats_display(total, added_so_far, skipped_so_far, 0)
            self.log(f"\n🎉 Historique terminé : {added_so_far} ajoutée(s), {skipped_so_far} ignorée(s) au total.")
            self.update_progress(100, "✅ Terminé")

        except Exception as e:
            self.log(f"❌ Erreur inattendue : {e}\n{traceback.format_exc()}")
            self.update_progress(0, "Erreur")
        finally:
            self.start_btn.configure(state="normal", text="🚀 Lancer / Reprendre")
            self.retry_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled", text="⏸️ Arrêter")
            self.app_lock.release("history")


# ============================================================
# FENÊTRE PRINCIPALE
# ============================================================

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Notion Video AI — Gemini")
        self.geometry("740x720")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        ctk.CTkLabel(self, text="🎬 Notion Video AI", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(15, 10))

        app_lock = AppLock()
        tabview = ctk.CTkTabview(self, width=720, height=640)
        tabview.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        tab1 = tabview.add("🎬 Ajouter une vidéo")
        tab2 = tabview.add("🔄 Migration Groq → Gemini")
        tab3 = tabview.add("📺 Historique des chaînes")

        AddVideoTab(tab1, app_lock).pack(fill="both", expand=True)
        MigrationTab(tab2, app_lock).pack(fill="both", expand=True)
        HistoryTab(tab3, app_lock).pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        if messagebox.askokcancel("Quitter", "Un traitement en cours sera interrompu proprement "
                                             "(le travail déjà effectué est sauvegardé). Quitter ?"):
            self.destroy()


if __name__ == "__main__":
    if STARTUP_ERROR:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Configuration manquante", STARTUP_ERROR)
    else:
        App().mainloop()