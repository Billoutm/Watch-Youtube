"""
watch_youtube.py — Surveillance quotidienne de chaînes YouTube.

Destiné à tourner une fois par jour via GitHub Actions (voir workflow fourni à côté).
Détecte les nouvelles vidéos via le flux RSS public de chaque chaîne (gratuit, zéro
scraping, zéro quota), demande à Gemini un avis rapide ("vaut le coup ou pas"), et ne
lance le pipeline complet (téléchargement + analyse vidéo + ajout Notion) que pour les
vidéos jugées intéressantes.

Réutilise directement les fonctions de main.py (même logique que l'onglet "Ajouter une
vidéo", sans interface graphique).
"""

import os
import sys
import time
import threading
import xml.etree.ElementTree as ET
import requests

# Réutilise tout le moteur déjà construit dans main.py (Notion, cache, Gemini, etc.)
from main import (
    DATABASE_ID, GEMINI_MODEL, LANG_INSTRUCTION,
    CHUNK_DURATION_WITH_CAPTIONS, CHUNK_DURATION_NO_CAPTIONS,
    MEDIA_RESOLUTION_WITH_CAPTIONS, MEDIA_RESOLUTION_NO_CAPTIONS,
    PROP_TITLE, PROP_PRICE, PROP_SOURCE_1, PROP_TYPE, PROP_GEMINI, DEFAULT_CATEGORY,
    QuotaExceededError, ServiceOverloadedError, StopRequested,
    check_dependencies, get_data_source_id, find_page_by_source_url,
    get_video_metadata, try_fetch_captions, download_video_cached,
    get_video_duration, estimate_chunk_count, analyze_full_video_cached,
    process_one_subject, apply_primary_subject_to_page, clear_work,
    get_work_id_for_url, call_gemini, notion_request,
)

CHANNELS_DATABASE_ID = os.getenv("CHANNELS_DATABASE_ID")
MAX_AUTO_DURATION_MINUTES = 90  # au-delà, on crée juste une page "en attente" dans la base existante
MAX_NEW_VIDEOS_PER_CHANNEL_PER_RUN = 10  # garde-fou si une chaîne a posté énormément d'un coup

YT_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}


def log(msg):
    print(msg, flush=True)


# ============================================================
# NOTION : base "Chaînes YouTube"
# ============================================================

def list_watched_channels(channels_ds_id):
    pages = []
    payload = {"page_size": 100}
    while True:
        response = notion_request("POST", f"https://api.notion.com/v1/data_sources/{channels_ds_id}/query",
                                  log, json=payload)
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
        last_video = props.get("Dernière vidéo", {}).get("rich_text", [])
        last_video_id = last_video[0]["text"]["content"] if last_video else None
        active = bool(props.get("Actif", {}).get("checkbox"))
        channels.append({
            "page_id": page["id"], "name": name, "channel_url": channel_url,
            "last_video_id": last_video_id, "active": active
        })
    return channels


def update_channel_last_seen(page_id, video_id):
    payload = {"properties": {"Dernière vidéo": {"rich_text": [{"text": {"content": video_id}}]}}}
    response = notion_request("PATCH", f"https://api.notion.com/v1/pages/{page_id}", log, json=payload)
    response.raise_for_status()


def create_placeholder_page(video_url, video_title, data_source_id):
    """Crée une page minimale, non traitée (Gemini décoché), pour une vidéo trop longue.
    L'onglet Migration la ramassera automatiquement au prochain lancement, exactement comme
    une page de l'ère Groq pas encore migrée — aucune base séparée nécessaire."""
    payload = {
        "parent": {"type": "data_source_id", "data_source_id": data_source_id},
        "properties": {
            PROP_TITLE: {"title": [{"text": {"content": f"⏳ {video_title[:90]}"}}]},
            PROP_PRICE: {"select": {"name": "Gratuite"}},
            PROP_SOURCE_1: {"url": video_url},
            PROP_TYPE: {"select": {"name": DEFAULT_CATEGORY}},
            PROP_GEMINI: {"checkbox": False},
        },
        "children": [{
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {
                "content": "⏳ Vidéo trop longue pour le traitement automatique quotidien. "
                           "Sera analysée au prochain lancement de l'onglet Migration."
            }}]}
        }]
    }
    try:
        response = notion_request("POST", "https://api.notion.com/v1/pages", log, json=payload)
        response.raise_for_status()
        log(f"   📥 Page 'en attente' créée pour migration manuelle : {video_title}")
    except Exception as e:
        log(f"   ⚠️ Impossible de créer la page en attente : {e}")


# ============================================================
# RÉSOLUTION ID DE CHAÎNE + FLUX RSS
# ============================================================

def get_channel_id(channel_url):
    """Résout un ID de chaîne (UCxxxx) à partir de n'importe quel lien YouTube
    (handle @nom, URL /channel/, /c/, /user/...)."""
    import re
    import subprocess
    import json as _json

    m = re.search(r"/channel/(UC[0-9A-Za-z_-]{22})", channel_url or "")
    if m:
        return m.group(1)

    try:
        result = subprocess.run(
            ["yt-dlp", "-J", "--flat-playlist", "--playlist-items", "1", channel_url],
            capture_output=True, text=True, check=True, timeout=30
        )
        data = _json.loads(result.stdout)
        return data.get("channel_id") or data.get("uploader_id")
    except Exception as e:
        log(f"   ⚠️ Impossible de résoudre l'ID de la chaîne pour {channel_url} : {e}")
        return None


def fetch_channel_feed(channel_id):
    """Flux RSS public YouTube : gratuit, officiel, zéro quota. Renvoie les entrées
    du plus récent au plus ancien (comportement standard de ce flux)."""
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as e:
        log(f"   ⚠️ Erreur lecture flux RSS : {e}")
        return []

    entries = []
    for entry in root.findall("atom:entry", YT_NS):
        video_id_el = entry.find("yt:videoId", YT_NS)
        title_el = entry.find("atom:title", YT_NS)
        link_el = entry.find("atom:link", YT_NS)
        if video_id_el is None or title_el is None or link_el is None:
            continue

        description = ""
        media_group = entry.find("media:group", YT_NS)
        if media_group is not None:
            desc_el = media_group.find("media:description", YT_NS)
            if desc_el is not None and desc_el.text:
                description = desc_el.text

        entries.append({
            "video_id": video_id_el.text,
            "title": title_el.text or "",
            "description": description,
            "url": link_el.get("href"),
        })
    return entries


# ============================================================
# TRI GEMINI ("vaut le coup ou pas")
# ============================================================

def triage_video(title, description, stop_event):
    """Appel texte seul, très léger : décide si la vidéo mérite le pipeline complet.
    Filtre aussi par logiciel de montage : seul DaVinci Resolve (ou aucun logiciel précis
    mentionné) passe, pour ne pas consommer de quota sur Premiere Pro / Final Cut / Avid / etc."""
    prompt = f"""Tu aides à filtrer automatiquement des vidéos YouTube pour un projet personnel
d'archivage de connaissances sur l'audiovisuel (montage, étalonnage, son, effets, storytelling, etc.).
{LANG_INSTRUCTION}

TITRE : {title}
DESCRIPTION : {description[:500]}

TÂCHE 1 (contenu) : cette vidéo a-t-elle un contenu éducatif/technique qui mérite d'être archivé et
analysé en détail (tutoriel, explication technique, retour d'expérience, comparatif...) ?
Réponds NON si c'est une simple annonce, un vlog sans contenu technique, une vidéo communautaire,
un live replay sans substance, ou du contenu publicitaire/sponsorisé pur.

TÂCHE 2 (logiciel) : seul DaVinci Resolve est utilisé par la personne qui archive. Si le titre ou
la description mentionne clairement un AUTRE logiciel de montage précis (Premiere Pro, Final Cut
Pro, Avid Media Composer, CapCut, Vegas Pro, etc.) sans mentionner Resolve, cette vidéo n'est PAS
pertinente, même si elle est par ailleurs intéressante. Si aucun logiciel précis n'est mentionné,
ou si Resolve est mentionné, c'est pertinent (les concepts génériques s'appliquent à tous les
logiciels).

Réponds UNIQUEMENT en JSON, sans aucun texte avant ou après :
{{"vaut_le_coup": true ou false, "raison": "explication très courte", "autre_logiciel_detecte": "nom du logiciel si un AUTRE logiciel précis a été détecté, sinon chaîne vide"}}"""

    # Les erreurs de quota/surcharge ne sont PAS absorbées ici : on veut arrêter tout le run
    # plutôt que de continuer à consommer du quota dans un état dégradé.
    data = call_gemini(prompt, log, stop_event, max_output_tokens=200, temperature=0.1)
    if not data:
        return True, "Tri impossible (réponse invalide), traitée par défaut par précaution"

    autre_logiciel = (data.get("autre_logiciel_detecte") or "").strip()
    if autre_logiciel:
        return False, f"Logiciel non pertinent détecté : {autre_logiciel} (seul DaVinci Resolve est suivi)"

    return bool(data.get("vaut_le_coup", True)), data.get("raison", "")


# ============================================================
# PIPELINE COMPLET POUR UNE VIDÉO (identique à l'onglet "Ajouter une vidéo", sans GUI)
# ============================================================

def process_video_automatically(video_url, data_source_id, stop_event):
    work_id = get_work_id_for_url(video_url)

    existing = find_page_by_source_url(data_source_id, video_url, log)
    if existing:
        log(f"   ℹ️ Déjà présente dans Notion : « {existing['title']} », ignorée.")
        return

    metadata = get_video_metadata(video_url, log)
    captions_text = try_fetch_captions(work_id, video_url, log)
    if captions_text:
        chunk_duration, media_resolution = CHUNK_DURATION_WITH_CAPTIONS, MEDIA_RESOLUTION_WITH_CAPTIONS
    else:
        chunk_duration, media_resolution = CHUNK_DURATION_NO_CAPTIONS, MEDIA_RESOLUTION_NO_CAPTIONS

    video_path, video_title = download_video_cached(work_id, video_url, metadata.get("title") or "Vidéo", log)
    if not video_path:
        log("   ❌ Échec du téléchargement, vidéo ignorée pour cette fois.")
        return

    duration = get_video_duration(video_path)
    nb_chunks = estimate_chunk_count(duration, chunk_duration)

    if duration and duration > MAX_AUTO_DURATION_MINUTES * 60:
        log(f"   ⏸️ Vidéo très longue ({int(duration // 60)} min) : page 'en attente' créée plutôt que traitée automatiquement.")
        create_placeholder_page(video_url, video_title, data_source_id)
        clear_work(work_id)
        return

    log(f"   🤖 Analyse Gemini ({nb_chunks} tranche(s))...")
    sujets, transcription_propre = analyze_full_video_cached(
        work_id, video_path, video_title, metadata.get("description", ""), captions_text,
        chunk_duration, media_resolution, log, stop_event
    )

    for subject in sujets:
        process_one_subject(video_url, subject, transcription_propre, data_source_id, log, stop_event)

    clear_work(work_id)


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

def run():
    stop_event = threading.Event()  # jamais déclenché en CI, juste pour réutiliser les fonctions partagées

    if not CHANNELS_DATABASE_ID:
        log("❌ CHANNELS_DATABASE_ID manquant dans les variables d'environnement.")
        sys.exit(1)
    if not check_dependencies(log):
        sys.exit(1)

    data_source_id = get_data_source_id(DATABASE_ID, log)
    channels_ds_id = get_data_source_id(CHANNELS_DATABASE_ID, log)
    channels = list_watched_channels(channels_ds_id)

    active_channels = [c for c in channels if c["active"] and c["channel_url"]]
    log(f"📺 {len(active_channels)} chaîne(s) active(s) sur {len(channels)} au total.")

    for channel in active_channels:
        log(f"\n🔍 Chaîne : {channel['name']}")
        channel_id = get_channel_id(channel["channel_url"])
        if not channel_id:
            log("   ❌ ID de chaîne introuvable, chaîne ignorée pour cette fois.")
            continue

        entries = fetch_channel_feed(channel_id)
        if not entries:
            log("   ℹ️ Aucune vidéo trouvée dans le flux.")
            continue

        if channel["last_video_id"] is None:
            # Première surveillance de cette chaîne : on ne traite PAS le passif,
            # on prend juste la vidéo la plus récente comme point de départ.
            log(f"   📌 Chaîne nouvellement surveillée, initialisation sans traiter l'historique.")
            update_channel_last_seen(channel["page_id"], entries[0]["video_id"])
            continue

        seen_ids = {c["video_id"] for c in channels} if False else None  # (placeholder, non utilisé)
        new_entries = []
        for entry in entries:
            if entry["video_id"] == channel["last_video_id"]:
                break
            new_entries.append(entry)
        new_entries = new_entries[:MAX_NEW_VIDEOS_PER_CHANNEL_PER_RUN]

        if not new_entries:
            log("   ✅ Rien de nouveau.")
            continue

        log(f"   🆕 {len(new_entries)} nouvelle(s) vidéo(s) détectée(s).")

        # Traitement de la plus ancienne à la plus récente, avec mise à jour immédiate
        # après chaque vidéo : une interruption ne fait jamais retraiter ce qui est déjà fait.
        for entry in reversed(new_entries):
            log(f"\n   ▶️ {entry['title']}")
            try:
                worth_it, reason = triage_video(entry["title"], entry["description"], stop_event)
                if worth_it:
                    log(f"      ✅ Retenue : {reason}")
                    process_video_automatically(entry["url"], data_source_id, stop_event)
                else:
                    log(f"      ⏭️ Ignorée : {reason}")
                update_channel_last_seen(channel["page_id"], entry["video_id"])

            except (QuotaExceededError, ServiceOverloadedError) as e:
                log(f"\n⏳ Arrêt de la surveillance pour aujourd'hui : {e}")
                log("💡 Cette vidéo sera retentée au prochain passage (pas marquée comme vue).")
                return  # on arrête tout le run, sans marquer cette vidéo comme vue

            except Exception as e:
                log(f"      ❌ Erreur sur cette vidéo : {e}")
                # on marque quand même comme vue pour ne pas boucler indéfiniment dessus
                update_channel_last_seen(channel["page_id"], entry["video_id"])

        time.sleep(2)

    log("\n🎉 Surveillance terminée pour aujourd'hui.")


if __name__ == "__main__":
    run()