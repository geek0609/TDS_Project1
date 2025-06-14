import json, time
from datetime import datetime, timezone
from pathlib import Path

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 4, 30, 23, 59, 59, tzinfo=timezone.utc)
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def launch_browser():
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    try:
        driver = webdriver.Chrome(options=opts)
    except Exception as e:
        print(f"Chrome failed: {e}")
        print("Trying Edge instead...")
        driver = webdriver.Edge()
    driver.get("https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34")
    print("Please complete the login in the opened browser window. Press Enter here once done.")
    input()
    return driver


def build_requests_session(driver):
    session = requests.Session()
    for c in driver.get_cookies():
        session.cookies.set(c["name"], c["value"])
    return session


def within_range(ts_str: str) -> bool:
    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return START_DATE <= ts <= END_DATE


def scrape_topic_list(session):
    ids = []
    page = 1
    while True:
        url = f"https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34/l/latest.json?page={page}"
        r = session.get(url)
        if r.status_code != 200:
            print(f"Stopping at page {page}, status code {r.status_code}")
            break
        data = r.json()
        (RAW_DIR / f"latest-{page}.json").write_text(json.dumps(data, indent=2))
        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            break
        print(f"Page {page}: {len(topics)} topics")
        for t in topics:
            created = t.get("created_at")
            if created and within_range(created):
                ids.append(t["id"])
        # stop if all topics on this page are older than start date
        if all(datetime.fromisoformat(t["created_at"].replace("Z", "+00:00")) < START_DATE for t in topics):
            break
        page += 1
        time.sleep(1)
    return ids


def scrape_topics(session, ids):
    out = []
    for tid in ids:
        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{tid}.json"
        r = session.get(url)
        if r.status_code != 200:
            print(f"Failed topic {tid}: {r.status_code}")
            continue
        data = r.json()
        (RAW_DIR / f"topic-{tid}.json").write_text(json.dumps(data, indent=2))
        out.append({
            "id": data.get("id"),
            "title": data.get("title"),
            "created_at": data.get("created_at"),
            "posts_count": data.get("posts_count"),
            "views": data.get("views"),
        })
        time.sleep(0.5)
    (BASE_DIR / "topics_summary.json").write_text(json.dumps(out, indent=2))


def main():
    driver = launch_browser()
    session = build_requests_session(driver)
    ids = scrape_topic_list(session)
    print(f"Total topics in range: {len(ids)}")
    scrape_topics(session, ids)
    print("Finished scraping.")
    driver.quit()


if __name__ == "__main__":
    main() 