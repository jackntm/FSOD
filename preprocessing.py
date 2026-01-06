import os
import cv2
import csv
import numpy as np

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#Tự tìm folder 'samples' nằm ngay cạnh file code
INPUT_ROOT_DIR = os.path.join(BASE_DIR, "samples")
OUTPUT_ROOT_DIR = os.path.join(BASE_DIR, "datasetTrain")

# Ngưỡng cắt cảnh (Thay đổi tùy video)
# Vì tự tính toán nên ngưỡng này khác với PySceneDetect.
# Gợi ý: 30.0 (Trung bình cộng độ lệch các pixel màu)
DIFF_THRESHOLD = 5

# Ngưỡng tối thiểu số frame của 1 cảnh (để tránh cắt vụn do nhiễu 1-2 frame)
MIN_SCENE_FRAMES = 50

def calculate_frame_diff(frame1, frame2):
    """
    Tính độ lệch giữa 2 frame liên tiếp.
    Sử dụng trung bình cộng độ lệch tuyệt đối (Mean Absolute Difference).
    """
    # Chuyển xám để tính toán nhanh hơn
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Tính hiệu tuyệt đối giữa các pixel
    diff = cv2.absdiff(g1, g2)
    
    # Tính giá trị trung bình của sự khác biệt
    non_zero_count = np.count_nonzero(diff)
    if non_zero_count == 0:
        return 0.0
    
    score = np.mean(diff)
    return score

def save_images_for_scene(video_path, scene_info, output_dir, video_name, scene_idx):
    """
    Lưu 3 ảnh (Đầu, Giữa, Cuối) cho cảnh đã cắt được
    """
    start_f = scene_info['start_frame']
    end_f = scene_info['end_frame']
    middle_f = int((start_f + end_f) / 2)
    
    cap = cv2.VideoCapture(video_path)
    
    save_list = [
        (start_f, "01"),
        (middle_f, "02"),
        (end_f, "03")
    ]
    
    scene_num_str = f"{scene_idx:03d}"
    saved_count = 0
    
    for f_idx, suffix in save_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            filename = f"{video_name}-Scene-{scene_num_str}-{suffix}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
    cap.release()
    return saved_count

def process_video_manual(video_path, output_dir, video_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" [ERR] Không mở được video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Danh sách lưu các cảnh tìm được
    # Format: {'scene_num': 1, 'start_frame': 0, 'end_frame': 100, ...}
    scenes = []
    
    # Biến theo dõi quá trình duyệt
    current_scene_start = 0
    prev_frame = None
    frame_idx = 0
    
    print(f" -> Đang quét {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Frame đầu tiên của video luôn là bắt đầu cảnh 1
        if frame_idx == 0:
            prev_frame = frame
            frame_idx += 1
            continue
        
        diff_score = calculate_frame_diff(prev_frame, frame)
        
        # Kiểm tra điều kiện cắt cảnh
        # 1. Độ lệch lớn hơn ngưỡng
        # 2. Cảnh hiện tại đủ dài (lớn hơn MIN_SCENE_FRAMES)
        if diff_score > DIFF_THRESHOLD and (frame_idx - current_scene_start) > MIN_SCENE_FRAMES:
            scenes.append({
                'start_frame': current_scene_start,
                'end_frame': frame_idx - 1,
                'fps': fps
            })
            
            # Bắt đầu cảnh mới tại frame hiện tại
            current_scene_start = frame_idx
            # print(f"    [Cut] Tại frame {frame_idx}. Diff: {diff_score:.2f}")
        
        # Cập nhật frame tham chiếu cho vòng lặp sau
        prev_frame = frame
        frame_idx += 1

    # Lưu cảnh cuối cùng (từ lần cắt cuối đến hết video)
    scenes.append({
        'start_frame': current_scene_start,
        'end_frame': frame_idx - 1,
        'fps': fps
    })
    
    cap.release()
    
    # --- XỬ LÝ KẾT QUẢ (Lưu CSV và Ảnh) ---
    
    # 1. Lưu CSV
    csv_filename = f"{video_name}_scenes.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    headers = ["Scene Number", "Start Frame", "End Frame", "Start Time(s)", "End Time(s)", "Duration(s)"]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        total_imgs = 0
        for i, scene in enumerate(scenes):
            s_frame = scene['start_frame']
            e_frame = scene['end_frame']
            fps_val = scene['fps'] if scene['fps'] > 0 else 25.0
            
            duration_frames = e_frame - s_frame
            start_sec = s_frame / fps_val
            end_sec = e_frame / fps_val
            dur_sec = duration_frames / fps_val
            
            writer.writerow([i+1, s_frame, e_frame, f"{start_sec:.3f}", f"{end_sec:.3f}", f"{dur_sec:.3f}"])
            
            # 2. Lưu Ảnh (gọi hàm save_images_for_scene)
            saved = save_images_for_scene(video_path, scene, output_dir, video_name, i+1)
            total_imgs += saved

    print(f" -> Hoàn tất: {len(scenes)} cảnh, {total_imgs} ảnh.")

def main_process():
    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.makedirs(OUTPUT_ROOT_DIR)
        
    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]
                parent = os.path.basename(root)
                
                output_folder = f"{parent}_SCENES"
                output_dir = os.path.join(OUTPUT_ROOT_DIR, output_folder)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                print(f"\n--- Xử lý: {file} ---")
                try:
                    process_video_manual(video_path, output_dir, video_name)
                except Exception as e:
                    print(f"LỖI: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    main_process()