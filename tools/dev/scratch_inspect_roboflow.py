import requests
import json
import re

url = "https://universe.roboflow.com/birdman-cmkcn/eg-student-cheating-detection"
res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
print("Status:", res.status_code)
text = res.text

# Look for version URLs
versions = set(re.findall(r"/birdman-cmkcn/eg-student-cheating-detection/dataset/\d+", text))
print("Versions:", versions)

# Search for class labels
matches = re.findall(r'"name":"([^"]+)"', text)
print("Names in json:", [m for m in matches if m in ["phone", "normal", "looking around", "cheating"]])

# Search for roboflow json data
match_json = re.search(r'<script id="__NEXT_DATA__" type="application/json">({.*?})</script>', text, re.DOTALL)
if match_json:
    data = json.loads(match_json.group(1))
    props = data.get("props", {}).get("pageProps", {})
    print("Project info keys:", list(props.keys()))
    project = props.get("project", {})
    print("Project name:", project.get("name"))
    print("Project classes:", project.get("classes"))
    print("Versions count:", len(project.get("versions", [])))
    for v in project.get("versions", []):
        print("  Version:", v.get("version"), "Images:", v.get("images"), "Classes:", v.get("classes"))
