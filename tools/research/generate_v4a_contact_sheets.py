import os
import glob
import json
import zipfile
import cv2
import numpy as np

def make_montage(images, grid_cols=10, tile_size=(128, 128)):
    """Create a contact sheet grid image from a list of images."""
    if not images:
        return None
    tiles = [cv2.resize(im, tile_size) for im in images]
    n = len(tiles)
    rows = (n + grid_cols - 1) // grid_cols
    # Pad with black tiles if necessary
    pad_count = rows * grid_cols - n
    for _ in range(pad_count):
        tiles.append(np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8))
    
    row_imgs = []
    for r in range(rows):
        row_imgs.append(np.hstack(tiles[r * grid_cols : (r + 1) * grid_cols]))
    return np.vstack(row_imgs)

def audit_scbehavior():
    print("Auditing SCBehavior High-Res crops...")
    coco_p = 'datasets/raw_v4/scbehavior_highres/repo/SCBehavior_COCO/coco/annotations/instances_train2017.json'
    img_dir = 'datasets/raw_v4/scbehavior_highres/repo/SCBehavior_COCO/coco/train2017'
    with open(coco_p, 'r') as f:
        data = json.load(f)
    
    img_map = {im['id']: (im['file_name'], im['width'], im['height']) for im in data['images']}
    cat_map = {c['id']: c['name'] for c in data['categories']}
    
    # Target categories to extract
    target_cats = {
        'read': ('scbehavior/read', 50),
        'write': ('scbehavior/write', 50),
        'turn_head': ('scbehavior/turn_head', 50),
        'lookup': ('scbehavior/lookup_normal', 50),
        'discuss': ('scbehavior/discuss', 50),
        'stand': ('scbehavior/stand', 50)
    }
    
    extracted = {k: [] for k in target_cats}
    
    # Cache loaded images to avoid reloading same image repeatedly
    loaded_imgs = {}
    
    for ann in data['annotations']:
        cat_name = cat_map[ann['category_id']]
        if cat_name in target_cats and len(extracted[cat_name]) < target_cats[cat_name][1]:
            im_file, im_w, im_h = img_map[ann['image_id']]
            if im_file not in loaded_imgs:
                p = os.path.join(img_dir, im_file)
                loaded_imgs[im_file] = cv2.imread(p)
            full_img = loaded_imgs[im_file]
            if full_img is None:
                continue
            x, y, w, h = [int(v) for v in ann['bbox']]
            crop = full_img[max(0, y):min(full_img.shape[0], y+h), max(0, x):min(full_img.shape[1], x+w)]
            if crop.size > 0:
                extracted[cat_name].append(crop)
                # save single crop
                idx = len(extracted[cat_name])
                out_p = os.path.join('reports/v4a/contact_sheets', target_cats[cat_name][0], f'sample_{idx:03d}.jpg')
                cv2.imwrite(out_p, crop)
                
        # Check if all filled
        if all(len(extracted[k]) >= target_cats[k][1] for k in target_cats):
            break
            
    for cat_name, crops in extracted.items():
        subfolder = target_cats[cat_name][0]
        montage = make_montage(crops, grid_cols=10, tile_size=(128, 128))
        if montage is not None:
            montage_p = os.path.join('reports/v4a/contact_sheets', subfolder, 'contact_sheet.jpg')
            cv2.imwrite(montage_p, montage)
            print(f"  Saved {len(crops)} crops and contact sheet for {cat_name} -> {montage_p}")

def audit_eduaction():
    print("Auditing EduAction video clips...")
    base = 'datasets/raw_v4/other_candidates/eduaction'
    target_cats = {
        'sleeping': ('eduaction/sleeping', 50),
        'writing': ('eduaction/writing', 50),
        'lecture': ('eduaction/lecture_normal', 50),
        'talking': ('eduaction/talking', 50),
        'play_phone': ('eduaction/play_phone', 50)
    }
    
    for cat_name, (subfolder, count) in target_cats.items():
        mp4s = sorted(glob.glob(os.path.join(base, cat_name, '*.mp4')))
        crops = []
        for i, mp4 in enumerate(mp4s[:count]):
            cap = cv2.VideoCapture(mp4)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # seek to middle frame
            mid = max(0, n_frames // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                crops.append(frame)
                out_p = os.path.join('reports/v4a/contact_sheets', subfolder, f'sample_{i+1:03d}.jpg')
                cv2.imwrite(out_p, frame)
        montage = make_montage(crops, grid_cols=10, tile_size=(128, 128))
        if montage is not None:
            montage_p = os.path.join('reports/v4a/contact_sheets', subfolder, 'contact_sheet.jpg')
            cv2.imwrite(montage_p, montage)
            print(f"  Saved {len(crops)} crops and contact sheet for {cat_name} -> {montage_p}")

def audit_aflw2000():
    print("Auditing AFLW2000-3D pose crops...")
    zip_path = 'datasets/raw_v4/head_pose_aflw2000/test.data.zip'
    pose_path = 'datasets/raw_v4/head_pose_aflw2000/configs/AFLW2000-3D.pose.npy'
    if not os.path.exists(zip_path) or not os.path.exists(pose_path):
        print("  AFLW2000 files missing, skipping.")
        return
        
    yaws = np.load(pose_path)
    # yaws are in degrees
    with zipfile.ZipFile(zip_path, 'r') as z:
        lines = z.read('test.data/AFLW2000-3D_crop.list').decode('utf-8').strip().split('\n')
        
        extreme_crops = []
        frontal_crops = []
        
        for i, fname in enumerate(lines):
            yaw = yaws[i]
            img_path_in_zip = f"test.data/AFLW2000-3D_crop/{fname.strip()}"
            if abs(yaw) > 45 and len(extreme_crops) < 50:
                img_data = z.read(img_path_in_zip)
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    extreme_crops.append(img)
                    out_p = os.path.join('reports/v4a/contact_sheets/aflw2000/turn_head_extreme', f'sample_{len(extreme_crops):03d}.jpg')
                    cv2.imwrite(out_p, img)
            elif abs(yaw) < 15 and len(frontal_crops) < 50:
                img_data = z.read(img_path_in_zip)
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    frontal_crops.append(img)
                    out_p = os.path.join('reports/v4a/contact_sheets/aflw2000/frontal_normal', f'sample_{len(frontal_crops):03d}.jpg')
                    cv2.imwrite(out_p, img)
                    
            if len(extreme_crops) >= 50 and len(frontal_crops) >= 50:
                break
                
        m_ext = make_montage(extreme_crops, grid_cols=10, tile_size=(128, 128))
        if m_ext is not None:
            cv2.imwrite('reports/v4a/contact_sheets/aflw2000/turn_head_extreme/contact_sheet.jpg', m_ext)
            print(f"  Saved {len(extreme_crops)} crops for turn_head_extreme -> contact_sheet.jpg")
            
        m_fro = make_montage(frontal_crops, grid_cols=10, tile_size=(128, 128))
        if m_fro is not None:
            cv2.imwrite('reports/v4a/contact_sheets/aflw2000/frontal_normal/contact_sheet.jpg', m_fro)
            print(f"  Saved {len(frontal_crops)} crops for frontal_normal -> contact_sheet.jpg")

if __name__ == '__main__':
    audit_scbehavior()
    audit_eduaction()
    audit_aflw2000()
    print("All contact sheets and audit crops generated.")
