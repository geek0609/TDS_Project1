#!/usr/bin/env python3
"""
Process Discourse Data for TDS Virtual TA

1. Scans scripts/raw directory for topic-*.json files (exported from Discourse).
2. Builds a consolidated data/discourse_data.json file containing:
   - topics: {topic_id, title, url, full_content}
   - all_qa_pairs: Extracts simple Q&A pairs where staff answers follow student questions.

The script is VERBOSE: prints progress, counts, and missing fields.
Run: python scripts/process_discourse_data.py
"""

import json
import re
import html
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("scripts/raw")
OUTPUT_FILE = Path("data/discourse_data.json")

STAFF_USERS = {  # Extend as needed
    "anand": "Anand",
    "anand.s": "Anand",
    "anands": "Anand",
    "tds_staff": "Staff",
}

HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    if not text:
        return ""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_url(title: str, topic_id: int) -> str:
    slug = re.sub(r"[^a-z0-9\- ]", "", title.lower()).replace(" ", "-")
    return f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}"


def process_topic_file(path: Path) -> Dict:
    """Process a single Discourse topic JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    topic_id = data.get("topic_id") or data.get("id")
    title = data.get("title", f"Topic {topic_id}")
    url = build_url(title, topic_id)

    posts = data.get("post_stream", {}).get("posts", [])
    full_content_parts: List[str] = []
    qa_pairs: List[Dict] = []

    # Simple heuristic: treat first post as question, next staff post as answer
    first_post_text = None
    answer_found = False

    for post in posts:
        cooked = clean_html(post.get("cooked", ""))
        if not cooked:
            continue
        full_content_parts.append(cooked)

        if first_post_text is None:
            first_post_text = cooked
            continue

        if not answer_found and post.get("username", "").lower() in STAFF_USERS:
            qa_pairs.append({
                "question": first_post_text,
                "answer": cooked,
                "topic_id": topic_id,
                "topic_title": title,
                "topic_url": url
            })
            answer_found = True

    full_content = "\n\n".join(full_content_parts)

    topic_entry = {
        "topic_id": topic_id,
        "title": title,
        "url": url,
        "full_content": full_content
    }

    return {"topic": topic_entry, "qa_pairs": qa_pairs}


def main():
    if not RAW_DIR.exists():
        logger.error(f"{RAW_DIR} does not exist. Ensure raw Discourse JSON files are available.")
        return

    all_topics: List[Dict] = []
    all_qa_pairs: List[Dict] = []

    files = sorted(RAW_DIR.glob("topic-*.json"))
    logger.info(f"Found {len(files)} raw Discourse topic files to process.")

    for idx, file in enumerate(files, 1):
        try:
            res = process_topic_file(file)
            all_topics.append(res["topic"])
            all_qa_pairs.extend(res["qa_pairs"])
            if idx % 50 == 0:
                logger.info(f"Processed {idx}/{len(files)} topics...")
        except Exception as e:
            logger.warning(f"Error processing {file.name}: {e}")

    logger.info(f"\nFinished processing Discourse data:")
    logger.info(f"  Topics processed     : {len(all_topics)}")
    logger.info(f"  QA pairs extracted   : {len(all_qa_pairs)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "topics": all_topics,
            "all_qa_pairs": all_qa_pairs
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved consolidated data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main() 