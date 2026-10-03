import io
import struct
import zipfile
import requests

def get_remote_zip_namelist(url):
    # Get file size
    head = requests.head(url, allow_redirects=True)
    if head.status_code != 200:
        return f"Error: status {head.status_code}"
    size = int(head.headers.get("content-length", 0))
    if size == 0:
        return "Error: size 0"
    
    # Read the last 65KB of the file (contains EOCD record)
    read_size = min(size, 65536)
    headers = {"Range": f"bytes={size - read_size}-{size - 1}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code not in (200, 206):
        return f"Range error: {resp.status_code}"
    
    data = resp.content
    # Find End of Central Directory (EOCD) signature: 0x06054b50
    eocd_idx = data.rfind(b"\x50\x4b\x05\x06")
    if eocd_idx == -1:
        return "EOCD signature not found"
    
    # Parse EOCD
    # eocd: signature(4), disk_num(2), cd_disk(2), total_entries_disk(2), total_entries(2), cd_size(4), cd_offset(4), comment_len(2)
    eocd = data[eocd_idx:eocd_idx + 22]
    sig, disk, cd_disk, entries_disk, total_entries, cd_size, cd_offset, comment_len = struct.unpack("<IHHHHIIH", eocd)
    
    # Fetch Central Directory
    cd_headers = {"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"}
    cd_resp = requests.get(url, headers=cd_headers)
    if cd_resp.status_code not in (200, 206):
        return f"CD fetch error: {cd_resp.status_code}"
    
    cd_data = cd_resp.content
    # Parse filenames from Central Directory
    # Each CD entry starts with 0x02014b50
    filenames = []
    idx = 0
    while idx < len(cd_data):
        if cd_data[idx:idx+4] != b"\x50\x4b\x01\x02":
            break
        # signature(4), version_made(2), version_needed(2), flags(2), method(2), time(2), date(2), crc(4), csize(4), usize(4), name_len(2), extra_len(2), comment_len(2), ...
        entry = cd_data[idx:idx+46]
        if len(entry) < 46:
            break
        name_len, extra_len, comment_len = struct.unpack("<HHH", entry[28:34])
        name_bytes = cd_data[idx+46:idx+46+name_len]
        try:
            fname = name_bytes.decode("utf-8")
        except:
            fname = name_bytes.decode("gbk", errors="replace")
        filenames.append(fname)
        idx += 46 + name_len + extra_len + comment_len
    return filenames

if __name__ == "__main__":
    targets = [
        ("SCB5_Teacher", "https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2.zip"),
        ("SCB_LLM_read_write", "https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB_LLM_202506/%E8%AF%BB%E5%86%99.zip"),
        ("SCB_LLM_discuss", "https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB_LLM_202506/%E8%AE%A8%E8%AE%BA.zip"),
        ("SCB_LLM_listen", "https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB_LLM_202506/%E5%90%AC%E8%AE%B2.zip"),
        ("SCB_mosaic", "https://huggingface.co/datasets/wintonYF/SCB-public-mosaic/resolve/main/SCB-public-mosaic.zip"),
    ]
    for label, url in targets:
        print(f"=== Checking {label} ===")
        res = get_remote_zip_namelist(url)
        if isinstance(res, list):
            print(f"Total members: {len(res)}")
            sample = [f for f in res if not f.endswith('/')][:10]
            for s in sample:
                print("  ", s.encode("ascii", "replace").decode("ascii"))
        else:
            print("  Result:", res)
