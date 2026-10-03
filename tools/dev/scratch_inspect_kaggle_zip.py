import requests
import struct
import zlib

download_api_url = 'https://www.kaggle.com/api/v1/datasets/download/shreyasudaya/scb-05-dataset'
with requests.get(download_api_url, stream=True, allow_redirects=False) as res:
    signed_url = res.headers.get('Location')

head = requests.head(signed_url)
size = int(head.headers.get('content-length', 0))

resp = requests.get(signed_url, headers={'Range': f'bytes={size - 1000}-{size - 1}'})
loc_idx = resp.content.rfind(b'\x50\x4b\x06\x07')
loc = resp.content[loc_idx:loc_idx+20]
sig, disk_num, zip64_cd_offset, total_disks = struct.unpack('<IIQI', loc)

z64_resp = requests.get(signed_url, headers={'Range': f'bytes={zip64_cd_offset}-{zip64_cd_offset+100}'})
sig, rec_size, v1, v2, d1, d2, ent1, ent2, cd_size, real_cd_offset = struct.unpack('<IQHHIIQQQQ', z64_resp.content[:56])

yaml_entries = []
chunk_size = 2 * 1024 * 1024
offset = real_cd_offset
remaining = cd_size
buffer = b''

while remaining > 0:
    fetch = min(remaining, chunk_size)
    r = requests.get(signed_url, headers={'Range': f'bytes={offset}-{offset + fetch - 1}'})
    data = buffer + r.content
    offset += fetch
    remaining -= fetch
    
    idx = 0
    while idx < len(data) - 46:
        if data[idx:idx+4] != b'\x50\x4b\x01\x02':
            break
        nl, el, cl = struct.unpack('<HHH', data[idx+28:idx+34])
        if idx + 46 + nl > len(data):
            break
        # local header offset is at offset 42 (4 bytes)
        local_header_offset = struct.unpack('<I', data[idx+42:idx+46])[0]
        # compressed size at 20, uncompressed size at 24
        comp_size, uncomp_size = struct.unpack('<II', data[idx+20:idx+28])
        comp_method = struct.unpack('<H', data[idx+10:idx+12])[0]
        
        name = data[idx+46:idx+46+nl].decode('utf-8', errors='replace')
        if name.endswith('.yaml') or name.endswith('data.yaml'):
            yaml_entries.append((name, local_header_offset, comp_size, uncomp_size, comp_method))
        idx += 46 + nl + el + cl
    buffer = data[idx:]

print(f"Found {len(yaml_entries)} YAML files in SCB-05 archive:")
for name, lho, csize, usize, cmeth in yaml_entries:
    print(f"\n--- {name} (csize={csize}, usize={usize}) ---")
    # Fetch local file header + data
    # Local file header is 30 bytes + name_len + extra_len
    lh_resp = requests.get(signed_url, headers={'Range': f'bytes={lho}-{lho + 30 + 1000 + csize - 1}'})
    lh_data = lh_resp.content
    if lh_data[:4] == b'\x50\x4b\x03\x04':
        nl_lh, el_lh = struct.unpack('<HH', lh_data[26:30])
        file_data = lh_data[30 + nl_lh + el_lh : 30 + nl_lh + el_lh + csize]
        if cmeth == 8: # deflate
            content = zlib.decompress(file_data, -15)
        elif cmeth == 0:
            content = file_data
        else:
            content = b"Unknown compression"
        print(content.decode('utf-8', errors='replace'))
