import os
import time
import psutil
import json
from inference_core import run_video_inference

# ------------------------------------------------------------------
# [설정] 감지 경로 및 폴더명
# ------------------------------------------------------------------
# 1. SD카드 설정
SD_MOUNT_ROOT = ""
SD_TARGET_FOLDER = "DCIM/DJI_001"

# 2. 실시간 통신 폴더 설정 
STREAM_ROOT = "" 
STREAM_TARGET_FOLDER = "live_videos"

HISTORY_FILE = "processed_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_history(history_set):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(list(history_set), f)

def get_active_source():
    """
    현재 활성화된 영상 소스 경로를 찾습니다.
    우선순위 1: 실시간 통신 폴더 (STREAM_ROOT)
    우선순위 2: SD카드 마운트 경로 (SD_MOUNT_ROOT)
    """
    
    # 1. 실시간 통신 폴더 확인
    # 폴더가 존재하고, 안에 파일이 하나라도 있으면 이쪽을 우선시함
    stream_target = os.path.join(STREAM_ROOT, STREAM_TARGET_FOLDER)
    if os.path.exists(stream_target):
        if len(os.listdir(stream_target)) > 0:
            return "STREAM", stream_target

    # 2. SD카드 확인 (고정 경로 우선)
    if os.path.exists(SD_MOUNT_ROOT):
        return "SD_CARD", os.path.join(SD_MOUNT_ROOT, SD_TARGET_FOLDER)
    
    # 3. SD카드 자동 감지 (Fallback)
    base_media = "/media/inha-vl"
    if os.path.exists(base_media):
        partitions = psutil.disk_partitions()
        for p in partitions:
            if base_media in p.mountpoint:
                if "MISC" in p.mountpoint:
                    mount_point = os.path.dirname(p.mountpoint)
                else:
                    mount_point = p.mountpoint
                
                return "SD_CARD_AUTO", os.path.join(mount_point, SD_TARGET_FOLDER)

    return None, None

def main():
    print("=================================================")
    print("   AIROVISION MONITORING SYSTEM (Dual Mode)   ")
    print("   [Mode 1] Real-time Stream Folder Monitor      ")
    print("   [Mode 2] SD Card Auto-Mount Monitor           ")
    print("=================================================")
    
    processed_files = load_history()
    print(f"[System] Loaded {len(processed_files)} history records.")

    current_source_type = None

    try:
        while True:
            # 1. 활성 소스 감지 (Stream or SD)
            source_type, target_dir = get_active_source()
            
            if target_dir:
                # 소스가 변경되었을 때만 로그 출력
                if source_type != current_source_type:
                    print(f"\n[System] 🟢 Active Source Detected: {source_type}")
                    print(f"[System] 📂 Monitoring Path: {target_dir}")
                    current_source_type = source_type
                    time.sleep(2) # 안정화 대기

                if os.path.exists(target_dir):
                    files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
                    new_files_found = False
                    
                    for filename in files:
                        if filename not in processed_files:
                            print(f"\n[System] ✨ New Content Found: {filename}")
                            full_path = os.path.join(target_dir, filename)
                            
                            # ==============================
                            # AI 추론 실행
                            # ==============================
                            success = run_video_inference(full_path)
                            
                            if success:
                                print(f"[System] ✅ Process Complete: {filename}")
                                processed_files.add(filename)
                                save_history(processed_files)
                                new_files_found = True
                            else:
                                print(f"[System] ❌ Process Failed: {filename}")
                    
                    if not new_files_found:
                        print(f"\r[System] [{source_type}] Waiting for new files...", end="")
                else:
                    # 경로는 잡혔는데 폴더가 없는 경우
                    print(f"\r[System] Target folder not found: {target_dir}", end="")

            else:
                # 아무것도 연결 안 됨
                if current_source_type is not None:
                    print(f"\n[System] 🔴 Source Disconnected.")
                    current_source_type = None
                print("\r[System] Waiting for Connection (Stream/SD)...", end="")

            time.sleep(3)

    except KeyboardInterrupt:
        print("\n[System] Shutting down...")
        save_history(processed_files)

if __name__ == "__main__":

    main()
