import os
import requests
import json
import re

token = None
try:
    with open('token.txt', 'r') as f:
        token = f.read().strip()
except Exception as e:
    print(f"Failed to read token: {e}")

if not token:
    print("Could not find GitHub token in ~/.git-credentials.")
    exit(1)

repo = "vyroxat/vsynth"
release_url = f"https://api.github.com/repos/{repo}/releases"

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

data = {
    "tag_name": "v0.5.0-prerelease",
    "name": "VSynth v0.5.0 (Pre-Release)",
    "body": "Pre-release standalone executable containing the new Faster-Whisper + VAD engine for music-immune lyric transcription.",
    "draft": False,
    "prerelease": True
}

print("Creating release...")
response = requests.post(release_url, headers=headers, json=data)
if response.status_code != 201:
    print(f"Failed to create release: {response.status_code} {response.text}")
    # Maybe release tag already exists? Let's check releases and get the latest
    if "already_exists" in response.text:
        print("Release tag already exists. Fetching existing release...")
        rel_resp = requests.get(f"https://api.github.com/repos/{repo}/releases/tags/v0.5.0-prerelease", headers=headers)
        if rel_resp.status_code == 200:
            release_info = rel_resp.json()
        else:
            print("Failed to fetch existing release.")
            exit(1)
    else:
        exit(1)
else:
    release_info = response.json()

upload_url = release_info["upload_url"].split("{")[0]
print(f"Release ID: {release_info['id']}")

files_to_upload = [
    "dist/VSynth.exe",
    "dist/vsynth-python-bundle.zip"
]

for file_path in files_to_upload:
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        exit(1) # We want to fail if the EXE isn't there
        
    file_name = os.path.basename(file_path)
    print(f"Uploading {file_name}...")
    
    with open(file_path, "rb") as f:
        asset_response = requests.post(
            f"{upload_url}?name={file_name}",
            headers={
                **headers,
                "Content-Type": "application/octet-stream"
            },
            data=f
        )
    if asset_response.status_code == 201:
        print(f"Uploaded {file_name} successfully.")
    elif "already_exists" in asset_response.text:
        print(f"{file_name} already exists in release.")
    else:
        print(f"Failed to upload {file_name}: {asset_response.text}")
