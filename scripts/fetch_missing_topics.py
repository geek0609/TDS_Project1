import re
import requests
import os
from pathlib import Path

def fetch_missing_topics(test_file_path: str, raw_data_dir: str):
    """
    Parses a test file to find required discourse topic URLs,
    checks if they exist locally, and fetches them if they don't.
    """
    print("🚀 Starting to fetch missing discourse topics...")
    
    test_file = Path(test_file_path)
    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(exist_ok=True)

    if not test_file.exists():
        print(f"❌ Test file not found at {test_file_path}")
        return

    content = test_file.read_text()
    
    # Find all discourse topic URLs
    discourse_urls = re.findall(r'https://discourse\.onlinedegree\.iitm\.ac\.in/t/[^/]+/(\d+)', content)
    topic_ids = set(discourse_urls)
    
    if not topic_ids:
        print("🤷 No discourse topic IDs found in the test file.")
        return

    print(f"Found {len(topic_ids)} unique topic IDs required by tests: {', '.join(topic_ids)}")

    for topic_id in topic_ids:
        topic_file = raw_dir / f"topic-{topic_id}.json"
        
        if topic_file.exists():
            print(f"✅ Topic {topic_id} already exists locally.")
        else:
            print(f"⏳ Topic {topic_id} not found locally. Fetching...")
            url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}.json"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Check for valid JSON
                data = response.json()

                with open(topic_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(data, f, indent=2)
                
                print(f"✅ Successfully fetched and saved topic {topic_id}.")
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error fetching topic {topic_id}: {e}")
            except json.JSONDecodeError:
                print(f"❌ Error decoding JSON for topic {topic_id}. Content might not be valid JSON.")

if __name__ == "__main__":
    fetch_missing_topics("test_virtual_ta_api.py", "scripts/raw") 