import torch
import cv2
import numpy as np
import os
import json
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from typing import List, Tuple
from pathlib import Path
import csv
import pandas as pd
from rembg import remove

# Import các module AI
from transformers import AutoImageProcessor, AutoModel
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CẤU HÌNH ĐƯỜNG DẪN ---
DATASET_ROOT_DIR = os.path.join(BASE_DIR, "datasetTrain")
OBSERVING_ROOT_DIR = os.path.join(BASE_DIR, "samples")
PREDICT_ROOT_DIR = os.path.join(BASE_DIR, "datasetPredict")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "submission_results.json")


SAM2_CHECKPOINT = os.path.join(BASE_DIR, "models", "sam2.1_hiera_large.pt")
SAM2_CONFIG = os.path.join(BASE_DIR, "models", "sam2.1_hiera_l.yaml")

# --- HÀM HỖ TRỢ CHO SAM2 ---
def save_masked_objects(masks, image, base_filename, save_dir, scale_factor=1.0):
    count = 0
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    
    metadata = {} 
    metadata_path = os.path.join(save_dir, "crops_metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            try: metadata = json.load(f)
            except: metadata = {}

    for i, ann in enumerate(sorted_anns):
        bbox_resized = ann['bbox']
        x_r, y_r, w_r, h_r = [int(v) for v in bbox_resized]
        
        if w_r < 10 or h_r < 10: continue

        x_orig = int(x_r / scale_factor)
        y_orig = int(y_r / scale_factor)
        w_orig = int(w_r / scale_factor)
        h_orig = int(h_r / scale_factor)

        m = ann['segmentation']
        masked_image = np.zeros_like(image)
        masked_image[m] = image[m]
        
        y2_r = min(y_r + h_r, image.shape[0])
        x2_r = min(x_r + w_r, image.shape[1])
        y1_r = max(0, y_r)
        x1_r = max(0, x_r)
        
        crop_img = masked_image[y1_r:y2_r, x1_r:x2_r]
        
        obj_filename = f"{base_filename.replace('.jpg', '')}_obj_{i}.jpg"
        save_path = os.path.join(save_dir, obj_filename)
        
        if crop_img.size > 0:
            cv2.imwrite(save_path, crop_img)
            metadata[obj_filename] = {
                "x1": x_orig, "y1": y_orig, 
                "x2": x_orig + w_orig, "y2": y_orig + h_orig,
                "w": w_orig, "h": h_orig
            }
            count += 1
            
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    return count

# --- GIAI ĐOẠN 1: SAM2 (LƯU SANG DATASET PREDICT) ---
def run_sam2_pipeline():
    print("\n" + "="*60)
    print("GIAI ĐOẠN 1: CHẠY SAM2")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2 on {device.upper()}...")
    
    sam_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    checkpoint = torch.load(SAM2_CHECKPOINT, map_location=device)
    sam_model.load_state_dict(checkpoint['model'])
    sam_model.to(device).eval()
    
    mask_generator = SAM2AutomaticMaskGenerator(sam_model)
    print("SAM2 Ready!")

    total_crops = 0
    
    # Duyệt Input ở datasetTrain
    for root, dirs, files in os.walk(DATASET_ROOT_DIR):
        target_files = [f for f in files if f.endswith("-01.jpg")]
        if not target_files: continue
            
        folder_name = os.path.basename(root)
        print(f"\n>>> Đang xử lý input: {folder_name}")
        
        predict_folder_path = os.path.join(PREDICT_ROOT_DIR, folder_name)
        crops_dir = os.path.join(predict_folder_path, "crops_dinov2")
        
        if not os.path.exists(crops_dir):
            os.makedirs(crops_dir)

        for filename in target_files:
            image_path = os.path.join(root, filename)
            try:
                image = cv2.imread(image_path)
                if image is None: continue

                max_width = 1280
                scale = 1.0
                if image.shape[1] > max_width:
                    scale = max_width / image.shape[1]
                    width = int(image.shape[1] * scale)
                    height = int(image.shape[0] * scale)
                    image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                else:
                    image_resized = image

                imageRgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                masks = mask_generator.generate(imageRgb)
                
                num_saved = save_masked_objects(masks, image_resized, filename, crops_dir, scale_factor=scale)
                total_crops += num_saved
            except Exception as e:
                print(f"   [LỖI] {filename}: {e}")
                continue

    print(f"Hoàn tất SAM2. Tổng cộng: {total_crops} ảnh crop.")
    del mask_generator
    del sam_model
    del checkpoint
    torch.cuda.empty_cache()

# --- CÁC HÀM HỖ TRỢ DINOv2 ---
def remove_background(image: Image.Image) -> Image.Image:
    """
    Xóa nền - Phiên bản tự động xử lý import và lỗi GPU.
    """
    from io import BytesIO
    import rembg 
    
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=image.format or 'PNG') 
    img_data = img_byte_arr.getvalue()

    try:
        if hasattr(rembg, 'new_session'):
            session = rembg.new_session("u2net", providers=['CPUExecutionProvider'])
            result_bytes = rembg.remove(img_data, session=session)
        else:
            result_bytes = rembg.remove(img_data)
            
    except Exception as e:
        print(f"   [Debug Rembg] Fallback to default remove due to: {e}")
        result_bytes = rembg.remove(img_data)

    result_img = Image.open(BytesIO(result_bytes)).convert("RGB")
    return result_img

def extract_features(image, processor, model, device):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
        features = F.normalize(features, p=2, dim=1)
    return features

def load_images_from_directory(path, max_files=None):
    if not os.path.exists(path): return [], []
    image_files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if max_files: image_files = image_files[:max_files]
    images, names = [], []
    for filename in image_files:
        try:
            img_path = Path(path) / filename
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            names.append(filename)
        except Exception: pass
    return images, names

def find_top_objects_by_average(obj_feats, obj_names, ref_feats, top_k=15):
    """
    Tính điểm trung bình của mỗi object với TẤT CẢ các ref images.
    Trả về Top K object có điểm trung bình cao nhất.
    """
    if not obj_feats or not ref_feats:
        return []

    objs_tensor = torch.stack(obj_feats).squeeze() 
    refs_tensor = torch.stack(ref_feats).squeeze()
    
    if objs_tensor.dim() == 1: objs_tensor = objs_tensor.unsqueeze(0)
    if refs_tensor.dim() == 1: refs_tensor = refs_tensor.unsqueeze(0)

    sim_matrix = torch.mm(objs_tensor, refs_tensor.t())

    avg_scores = torch.mean(sim_matrix, dim=1) # [N]

    k_val = min(top_k, len(avg_scores))
    top_scores, top_indices = torch.topk(avg_scores, k=k_val)

    results = []
    for score, idx in zip(top_scores, top_indices):
        idx = idx.item()
        results.append({
            'match': obj_names[idx],
            'score': score.item(), 
        })
        
    return results

def visualize_top_matches(top_results, obj_names, obj_imgs, ref_imgs, output_dir, ref_label):
    """
    Vẽ 3 ảnh Ref ở hàng đầu.
    Vẽ Top K matches ở các hàng dưới (Grid 5 cột).
    """
    if not top_results: return

    name_to_img = {name: img for name, img in zip(obj_names, obj_imgs)}

    num_matches = len(top_results)
    cols = 5
    
    # Tính số hàng cần thiết: 
    # 1 hàng cho Ref + (số matches chia cho số cột, làm tròn lên)
    match_rows = (num_matches + cols - 1) // cols
    total_rows = 1 + match_rows 

    # Tạo hình kích thước lớn
    fig = plt.figure(figsize=(20, 4 * total_rows))
    plt.suptitle(f"Reference: {ref_label} - Top {num_matches} Matches", fontsize=16, fontweight='bold')
    
    for i, r_img in enumerate(ref_imgs):
        ax = plt.subplot(total_rows, cols, i + 1) 
        ax.imshow(r_img)
        ax.set_title(f"REF {i+1}", color='blue', fontweight='bold')
        ax.axis('off')

    for i, res in enumerate(top_results):
        match_name = res['match']
        score = res['score']
        
        if match_name in name_to_img:
            plot_idx = cols + i + 1
            
            ax = plt.subplot(total_rows, cols, plot_idx)
            ax.imshow(name_to_img[match_name])
            
            # Tiêu đề: Rank và Score
            ax.set_title(f"Rank #{i+1}\nScore: {score:.3f}", fontsize=10)

            if i == 0:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
            
            ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"visualize_top{num_matches}_{ref_label}.jpg")
    plt.savefig(save_path)
    plt.close(fig)

# --- GIAI ĐOẠN 2: DINOv2 (ĐỌC TỪ DATASET PREDICT) ---
def run_dinov2_pipeline():
    print("\n" + "="*60)
    print("GIAI ĐOẠN 2: DINOv2")
    print("="*60)

    if not os.path.exists(PREDICT_ROOT_DIR): 
        print("Chưa có datasetPredict, hãy chạy SAM2 trước.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading DINOv2 on {device}...")
    
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=True)
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    model.eval()
    
    subfolders = [f.path for f in os.scandir(PREDICT_ROOT_DIR) if f.is_dir()]
    
    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        video_id = folder_name.replace("_SCENES", "").replace("_SCENCE", "")
        crop_dir = os.path.join(folder_path, "crops_dinov2")
        if not os.path.exists(crop_dir): continue
            
        print(f"\nĐang xử lý: {folder_name}")
        
        ref_folder_name = folder_name.replace("_SCENES", "").replace("_SCENCE", "")
        ref_dir = os.path.join(OBSERVING_ROOT_DIR, ref_folder_name, "object_images")
        
        if not os.path.exists(ref_dir): 
            print(f"  Skipping: Không tìm thấy ảnh mẫu tại {ref_dir}")
            continue
        
        output_dir = folder_path 
            
        ref_imgs, ref_names = load_images_from_directory(ref_dir)
        obj_imgs, obj_names = load_images_from_directory(crop_dir)
        
        if not ref_imgs or not obj_imgs: 
            print("  Không đủ dữ liệu ảnh (ref hoặc crop trống).")
            continue
        
        # --- Xử lý xóa nền ảnh mẫu (Ref) ---
        ref_imgs_processed = []
        for img in ref_imgs:
            try:
                img_nobg = remove_background(img)
                ref_imgs_processed.append(img_nobg)
            except Exception as e:
                print(f"   [WARN] Lỗi xóa nền: {e}")
                ref_imgs_processed.append(img)

        # --- Extract Features ---
        ref_feats = [extract_features(img, processor, model, device) for img in ref_imgs_processed]

        obj_feats = [extract_features(img, processor, model, device) for img in obj_imgs]
        
        top_15_results = find_top_objects_by_average(obj_feats, obj_names, ref_feats, top_k=20)
        
        # Lưu kết quả
        json_out_path = os.path.join(output_dir, 'matching_results.json')
        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump(top_15_results, f, indent=4)
            
        print(f"  -> Đã lưu Top {len(top_15_results)} candidates.")
        
        if top_15_results:
            try:
                visualize_top_matches(
                    top_results=top_15_results,
                    obj_names=obj_names,
                    obj_imgs=obj_imgs,
                    ref_imgs=ref_imgs,
                    output_dir=output_dir,
                    ref_label=video_id
                )
                print(f"  -> Đã lưu ảnh visualize tại {output_dir}")
            except Exception as e:
                print(f"  [WARN] Lỗi visualize: {e}")


# --- GIAI ĐOẠN 3: TỔNG HỢP (INPUT PREDICT + CSV TRAIN) ---
def get_scene_frame_range(csv_path, scene_number):
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        row = df[df['Scene Number'] == scene_number]
        if not row.empty:
            return int(row.iloc[0]['Start Frame']), int(row.iloc[0]['End Frame'])
    except Exception as e:
        print(f"   [LỖI CSV] {csv_path}: {e}")
    return None, None

def parse_filename_info(filename):
    """
    Phân tích tên file để lấy thông tin Scene và Thứ tự ảnh.
    Input: "drone_video-Scene-170-01_obj_2.jpg"
    Output: (170, 1) -> (Scene ID, Image Sequence Number)
    """
    try:
        # Bỏ đuôi .jpg
        name = filename.rsplit('.', 1)[0]
        
        # Lấy phần gốc trước khi có _obj_
        if "_obj_" in name:
            base_name = name.split("_obj_")[0]
        else:
            base_name = name

        # Tách dấu gạch ngang "-"
        parts = base_name.split('-')
        
        # Tìm vị trí của từ "Scene" để định vị
        if "Scene" in parts:
            idx_scene_keyword = parts.index("Scene")
            
            # Scene ID thường nằm ngay sau chữ Scene
            scene_id = int(parts[idx_scene_keyword + 1])
            
            # Image Sequence (01, 02) thường nằm sau Scene ID
            # Kiểm tra xem phần tử tiếp theo có phải số không
            img_seq_str = parts[idx_scene_keyword + 2]
            img_seq = int(img_seq_str)
            
            return scene_id, img_seq
            
    except Exception as e:
        pass
    return None, None

def generate_submission_json():
    print("\n" + "="*60)
    print("GIAI ĐOẠN 3: TỔNG HỢP JSON (STITCHING - HÀN GẮN LIÊN TIẾP)")
    print("="*60)

    final_output = []
    if not os.path.exists(PREDICT_ROOT_DIR): return

    subfolders = [f.path for f in os.scandir(PREDICT_ROOT_DIR) if f.is_dir()]
    
    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        video_id = folder_name.replace("_SCENES", "").replace("_SCENCE", "")
        print(f"Đang xử lý: {video_id}...")

        # 1. Đọc matching & metadata
        matching_path = os.path.join(folder_path, "matching_results.json")
        if not os.path.exists(matching_path): continue
        with open(matching_path, 'r', encoding='utf-8') as f: matches = json.load(f)

        metadata_path = os.path.join(folder_path, "crops_dinov2", "crops_metadata.json")
        bbox_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f: bbox_metadata = json.load(f)

        # 2. Tìm CSV
        train_folder_path = os.path.join(DATASET_ROOT_DIR, folder_name)
        csv_path = None
        for root, dirs, files in os.walk(train_folder_path):
            for file in files:
                if file.endswith(".csv"):
                    csv_path = os.path.join(root, file); break
            if csv_path: break
        
        if not csv_path: continue

        # Bước 1: Tạo các mảnh annotation rời rạc 
        raw_annotations = []
        
        for match in matches:
            fname = match.get('match')
            
            if fname not in bbox_metadata: continue
            bbox = bbox_metadata[fname]

            scene_id, img_seq = parse_filename_info(fname)
            
            if scene_id is not None:

                start_f, end_f = get_scene_frame_range(csv_path, scene_id)
                
                if start_f is not None:

                    segment_bboxes = []
                    for f_idx in range(start_f, end_f + 1):
                        segment_bboxes.append({
                            "frame": int(f_idx),
                            "x1": int(bbox['x1']), "y1": int(bbox['y1']),
                            "x2": int(bbox['x2']), "y2": int(bbox['y2'])
                        })
                    
                    raw_annotations.append({
                        "bboxes": segment_bboxes
                    })

        # Bước 2: Thuật toán STITCHING
        if not raw_annotations: continue

        # 2.1. Sắp xếp các đoạn theo frame bắt đầu
        raw_annotations.sort(key=lambda x: x['bboxes'][0]['frame'])

        merged_annotations = []
        current_ann = raw_annotations[0]

        for i in range(1, len(raw_annotations)):
            next_ann = raw_annotations[i]
            
            curr_bboxes = current_ann['bboxes']
            next_bboxes = next_ann['bboxes']
            
            last_frame_curr = curr_bboxes[-1]['frame']
            first_frame_next = next_bboxes[0]['frame']
            
            if first_frame_next == last_frame_curr + 1:
                current_ann['bboxes'].extend(next_bboxes)
            elif first_frame_next <= last_frame_curr:
                pass       
            else:
                merged_annotations.append(current_ann)
                current_ann = next_ann
        
        # Lưu đoạn cuối cùng
        merged_annotations.append(current_ann)

        # Bước 3: Lưu kết quả
        video_data = {
            "video_id": video_id,
            "annotations": merged_annotations
        }
        final_output.append(video_data)
        print(f"  -> Đã gộp thành {len(merged_annotations)} chuỗi frame liên tục.")

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nHoàn tất! Kết quả đã được hàn gắn tại: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    run_sam2_pipeline()
    run_dinov2_pipeline()
    generate_submission_json()

