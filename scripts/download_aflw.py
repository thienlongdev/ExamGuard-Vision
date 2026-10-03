import os
import re
import urllib.request
import http.cookiejar

def download_gdrive_file(file_id, output_path):
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    resp = opener.open(url)
    content = resp.read()
    
    # Check if this is an HTML virus-scan confirmation page
    if b'drive.usercontent.google.com' in content or b'confirm=' in content or b'download_warning' in content:
        text = content.decode('utf-8', errors='ignore')
        # Try to find direct action form
        m_form = re.search(r'action="([^"]+)"', text)
        if m_form:
            action_url = m_form.group(1).replace('&amp;', '&')
            inputs = re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', text)
            query_str = urllib.parse.urlencode(inputs)
            download_url = f"{action_url}?{query_str}" if query_str else action_url
            print(f"Following GET to: {download_url[:100]}...")
            resp = opener.open(download_url)
        else:
            print("Could not find confirmation form action.")
            return False
            
    print(f"Downloading to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            print('.', end='', flush=True)
    print(f"\nFinished: {output_path} ({os.path.getsize(output_path)} bytes)")
    return True

if __name__ == '__main__':
    out = 'datasets/raw_v4/head_pose_aflw2000/test.data.zip'
    download_gdrive_file('1r_ciJ1M0BSRTwndIBt42GlPFRv6CvvEP', out)
