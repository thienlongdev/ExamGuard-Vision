import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')

hf_base = "https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main"

# Let's inspect the files in wintonYF/SCB-Dataset
res = requests.get('https://huggingface.co/api/datasets/wintonYF/SCB-Dataset')
data = res.json()

zip_files = []
for s in data.get('siblings', []):
    fname = s.get('rfilename')
    if fname.endswith('.zip') or fname.endswith('.yaml'):
        zip_files.append(fname)

print(f"Total zip/yaml files in wintonYF/SCB-Dataset: {len(zip_files)}")
for f in zip_files:
    head = requests.head(f"{hf_base}/{f}", allow_redirects=True)
    size_mb = int(head.headers.get('content-length', 0)) / (1024 * 1024)
    print(f"{f}: {size_mb:.2f} MB (status: {head.status_code})")

# Also check wintonYF/SCB-public-mosaic
res_mosaic = requests.get('https://huggingface.co/api/datasets/wintonYF/SCB-public-mosaic')
if res_mosaic.status_code == 200:
    for s in res_mosaic.json().get('siblings', []):
        fname = s.get('rfilename')
        head = requests.head(f"https://huggingface.co/datasets/wintonYF/SCB-public-mosaic/resolve/main/{fname}", allow_redirects=True)
        size_mb = int(head.headers.get('content-length', 0)) / (1024 * 1024)
        print(f"mosaic -> {fname}: {size_mb:.2f} MB")
