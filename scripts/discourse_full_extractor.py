#!/usr/bin/env python3
"""
TDS Discourse Category Scraper
==============================
Scrapes topics from the TDS Knowledge Base category within a specific date range.
Focused on the project requirements: Jan 1 - Apr 14, 2025.
"""

import json
import time
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

# TDS Knowledge Base category URL
TDS_CATEGORY_URL = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34"


def launch_browser():
    """Launch Chrome browser and navigate to TDS category for manual login."""
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    
    # Fix chromedriver path issue
    driver_path = ChromeDriverManager().install()
    if driver_path.endswith("THIRD_PARTY_NOTICES.chromedriver"):
        candidate = Path(driver_path).with_name("chromedriver.exe")
        if candidate.exists():
            driver_path = str(candidate)
    
    service = ChromeService(driver_path)
    driver = webdriver.Chrome(service=service, options=opts)
    
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
    except Exception:
        return False


def scrape_tds_topic_list(session):
    """Scrape topic IDs from TDS category pages."""
    topic_ids = []
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
            
            if not created_at:
                continue
                
            # Check if topic is too old (before our start date)
            topic_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if topic_date >= START_DATE:
                all_topics_too_old = False
                
            # Add topic if it's in our date range
            if within_date_range(created_at):
                topic_ids.append(topic["id"])
                page_has_valid_topics = True
                print(f"  Added topic {topic['id']}: {topic.get('title', 'No title')[:50]}...")
        
        # If all topics on this page are older than our start date, stop
        if all_topics_too_old:
            print("Reached topics older than start date, stopping pagination.")
            break
            
        page += 1
        time.sleep(1)  # Be nice to the server
    
    print(f"Found {len(topic_ids)} topics in date range")
    return topic_ids


def scrape_topic_details(session, topic_ids):
    """Download full JSON for each topic."""
    topic_summaries = []
    
    print(f"Downloading details for {len(topic_ids)} topics...")
    
    for i, topic_id in enumerate(topic_ids, 1):
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
    
    # Launch browser for manual login
    driver = launch_browser()
    
    try:
        # Create authenticated session
        session = build_requests_session(driver)
        
        # Get topic IDs from category pages
        topic_ids = scrape_tds_topic_list(session)
        
        if not topic_ids:
            print("❌ No topics found in date range!")
            return
            
        # Download full topic data
        summaries = scrape_topic_details(session, topic_ids)
        
        print("✅ Scraping completed successfully!")
        print(f"📊 Total topics scraped: {len(summaries)}")
        
    finally:
        driver.quit()


if __name__ == "__main__":
    main() 