#!/usr/bin/env python3
"""
TDS Discourse Category Scraper
==============================
Scrapes topics from the TDS Knowledge Base category within a specific date range,
and also ensures that any specific topics required by the test suite are fetched.
"""

import json
import time
import re
from datetime import datetime, timezone
from pathlib import Path
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager


# Date range from project spec
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 4, 14, 23, 59, 59, tzinfo=timezone.utc)

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
TEST_FILE_PATH = BASE_DIR.parent / "test_virtual_ta_api.py"


# TDS Knowledge Base category URL
TDS_CATEGORY_URL = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34"


def get_required_topic_ids_from_tests(test_file: Path) -> set[int]:
    """Parses the test file to find all mentioned discourse topic IDs."""
    if not test_file.exists():
        print(f"⚠️ Test file not found at {test_file}, cannot extract required topics.")
        return set()

    print(f"🔍 Reading test file to find required topic IDs: {test_file.name}")
    content = test_file.read_text(encoding="utf-8")
    
    # Regex to find all discourse topic URLs and extract their IDs
    found_ids = re.findall(r'https://discourse\.onlinedegree\.iitm\.ac\.in/t/[^/]+/(\d+)', content)
    
    if not found_ids:
        print("🤷 No required topic IDs found in the test file.")
        return set()
        
    topic_ids = {int(id) for id in found_ids}
    print(f"✅ Found {len(topic_ids)} required topic IDs from tests: {topic_ids}")
    return topic_ids


def launch_browser():
    """Launch Chrome browser and navigate to TDS category for manual login."""
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    
    # Fix chromedriver path issue
    try:
        driver_path = ChromeDriverManager().install()
    except Exception as e:
        print(f"❌ Could not download ChromeDriver: {e}")
        print("Please ensure you have a working internet connection.")
        return None

    if driver_path and driver_path.endswith("THIRD_PARTY_NOTICES.chromedriver"):
        candidate = Path(driver_path).with_name("chromedriver.exe")
        if candidate.exists():
            driver_path = str(candidate)
    
    try:
        service = ChromeService(driver_path)
        driver = webdriver.Chrome(service=service, options=opts)
    except Exception as e:
        print(f"❌ Failed to start Selenium Chrome driver: {e}")
        print("Ensure Chrome is installed and updated.")
        return None

    driver.get(TDS_CATEGORY_URL)
    print("Please complete the login in the opened browser window.")
    print("Press Enter here once you're logged in and can see the TDS category page.")
    input()
    return driver


def build_requests_session(driver):
    """Extract cookies from Selenium driver to create authenticated requests session."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    for cookie in driver.get_cookies():
        session.cookies.set(cookie["name"], cookie["value"])
    
    return session


def within_date_range(timestamp_str: str) -> bool:
    """Check if timestamp is within our target date range."""
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return START_DATE <= ts <= END_DATE
    except (ValueError, TypeError):
        return False


def scrape_tds_topic_list(session):
    """Scrape topic IDs from TDS category pages based on creation or last post date."""
    topic_ids = set()
    page = 0
    
    print(f"Scraping TDS topics from {START_DATE.date()} to {END_DATE.date()}...")
    
    while True:
        # Use the TDS category latest JSON endpoint
        url = f"https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34/l/latest.json?page={page}"
        
        print(f"Fetching page {page}...")
        response = session.get(url)
        
        if response.status_code != 200:
            print(f"Stopping at page {page}, status code {response.status_code}")
            break
            
        data = response.json()
        
        # Save raw page data
        (RAW_DIR / f"tds-latest-page-{page}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        
        topics = data.get("topic_list", {}).get("topics", [])
        
        if not topics:
            print(f"No topics found on page {page}, stopping.")
            break
            
        print(f"Page {page}: found {len(topics)} topics")
        
        page_has_valid_topics = False
        all_topics_too_old = True
        
        for topic in topics:
            created_at = topic.get("created_at")
            last_posted_at = topic.get("last_posted_at")

            # Check if topic creation or last post is too old (before our start date)
            # This helps us stop pagination early if we hit a page of old, inactive topics.
            created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")) if created_at else None
            if created_date and created_date >= START_DATE:
                all_topics_too_old = False
            
            # Add topic if its creation OR last post is within our date range
            if (created_at and within_date_range(created_at)) or \
               (last_posted_at and within_date_range(last_posted_at)):
                topic_ids.add(topic["id"])
                page_has_valid_topics = True
                print(f"  Added topic {topic['id']}: {topic.get('title', 'No title')[:50]}...")
        
        # If all topics on this page were CREATED before our start date, stop.
        # This is a heuristic to prevent scraping the entire forum history.
        if all_topics_too_old:
            print("Reached topics that were all created before start date, stopping pagination.")
            break
            
        page += 1
        time.sleep(1)  # Be nice to the server
    
    print(f"Found {len(topic_ids)} topics in date range from category scan.")
    return list(topic_ids)


def scrape_topic_details(session, topic_ids):
    """Download full JSON for each topic."""
    topic_summaries = []
    
    print(f"Downloading details for {len(topic_ids)} topics...")
    
    for i, topic_id in enumerate(sorted(list(topic_ids)), 1):
        # Check if topic already exists
        topic_file = RAW_DIR / f"topic-{topic_id}.json"
        if topic_file.exists():
            print(f"[{i}/{len(topic_ids)}] Skipping topic {topic_id}, already exists.")
            # Still add to summary from existing file
            try:
                topic_data = json.loads(topic_file.read_text(encoding="utf-8"))
                summary = {
                    "id": topic_data.get("id"),
                    "title": topic_data.get("title"),
                    "created_at": topic_data.get("created_at"),
                    "posts_count": topic_data.get("posts_count"),
                    "views": topic_data.get("views"),
                    "like_count": topic_data.get("like_count", 0),
                    "category_id": topic_data.get("category_id"),
                }
                topic_summaries.append(summary)
            except Exception:
                pass # Ignore if file is corrupted, it will be re-fetched next time
            continue

        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}.json"
        
        print(f"[{i}/{len(topic_ids)}] Fetching topic {topic_id}...")
        
        response = session.get(url)
        
        if response.status_code != 200:
            print(f"  Failed to fetch topic {topic_id}: HTTP {response.status_code}")
            continue
            
        try:
            topic_data = response.json()
            
            # Save full topic JSON
            (RAW_DIR / f"topic-{topic_id}.json").write_text(
                json.dumps(topic_data, indent=2), encoding="utf-8"
            )
            
            # Extract summary info
            summary = {
                "id": topic_data.get("id"),
                "title": topic_data.get("title"),
                "created_at": topic_data.get("created_at"),
                "posts_count": topic_data.get("posts_count"),
                "views": topic_data.get("views"),
                "like_count": topic_data.get("like_count", 0),
                "category_id": topic_data.get("category_id"),
            }
            topic_summaries.append(summary)
            
            print(f"  Saved: {topic_data.get('title', 'Untitled')[:50]}...")
            
        except Exception as e:
            print(f"  Error processing topic {topic_id}: {e}")
            
        time.sleep(0.5)  # Rate limiting
    
    # Save summary file
    summary_file = BASE_DIR / "topics_summary.json"
    summary_file.write_text(json.dumps(topic_summaries, indent=2), encoding="utf-8")
    
    print(f"Saved {len(topic_summaries)} topic summaries to {summary_file}")
    return topic_summaries


def main():
    """Main execution function."""
    print("🚀 Starting TDS Discourse scraper...")
    print(f"📅 Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"📁 Output directory: {RAW_DIR}")

    # 1. Get topic IDs required by the test suite
    required_topic_ids = get_required_topic_ids_from_tests(TEST_FILE_PATH)
    
    # 2. Launch browser for manual login
    driver = launch_browser()
    if not driver:
        return
    
    try:
        # 3. Create authenticated session
        session = build_requests_session(driver)
        
        # 4. Get topic IDs from the main TDS category pages
        category_topic_ids = scrape_tds_topic_list(session)
        
        # 5. Combine the topic lists
        final_topic_ids = set(category_topic_ids).union(required_topic_ids)
        
        if not final_topic_ids:
            print("❌ No topics found to download!")
            return
            
        print(f"\nTotal unique topics to download: {len(final_topic_ids)}")

        # 6. Download full topic data for all found topics
        summaries = scrape_topic_details(session, final_topic_ids)
        
        print("\n✅ Scraping completed successfully!")
        print(f"📊 Total topics processed: {len(summaries)}")
        
    finally:
        driver.quit()


if __name__ == "__main__":
    main() 