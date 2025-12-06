import numpy as np
import cv2
import os
import sys
import time
import requests
import json
import glob
from datetime import datetime, timedelta
from threading import Thread
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)
from gps_helper import find_latest_image_before_video, get_gps_from_image

# -----------------------------------------------------------------------------------------------
# [설정] 서버 정보 및 모델 경로
# -----------------------------------------------------------------------------------------------
HEF_PATH = ""

# [서버 설정]
SERVER_BASE_URL  = ""
URL_IMAGE_UPLOAD = f"{SERVER_BASE_URL}/upload-img"
URL_METADATA     = f"{SERVER_BASE_URL}/defect-info"

# [재전송 큐 설정]
RETRY_DIR = "failed_uploads"
os.makedirs(RETRY_DIR, exist_ok=True)

# [탐지 설정]
CONF_THRESHOLD = 0.45
IOU_THRESHOLD  = 0.45
NUM_CLASSES    = 1

# -----------------------------------------------------------------------------------------------
# [Helper] 재전송 로직 (큐 관리)
# -----------------------------------------------------------------------------------------------
def save_failed_request(frame, frame_count, timestamp, lat, lon):
    """
    전송 실패 시 이미지와 메타데이터 로컬 파일로 저장
    """
    try:
        base_name = f"retry_{int(time.time())}_{frame_count}"
        img_path = os.path.join(RETRY_DIR, f"{base_name}.jpg")
        meta_path = os.path.join(RETRY_DIR, f"{base_name}.json")

        cv2.imwrite(img_path, frame)

        meta_data = {
            "frame_count": frame_count,
            "timestamp": timestamp,
            "latitude": lat,
            "longitude": lon
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f)
            
        print(f"\n[Retry] 💾 Saved failed request to queue: {base_name}")

    except Exception as e:
        print(f"[Retry] ❌ Critical Error saving failure: {e}")

def process_retry_queue():
    """
    저장된 실패 파일들이 있다면, 빌 때까지 루프를 돌며 재전송을 시도합니다.
    """

    failed_files = sorted(glob.glob(os.path.join(RETRY_DIR, "*.json")))
    
    if not failed_files: return

    print(f"\n[Retry] 🔄 Processing {len(failed_files)} items in retry queue...")

    for meta_path in failed_files:
        img_path = meta_path.replace(".json", ".jpg")
        
        if not os.path.exists(img_path):
            os.remove(meta_path)
            continue

        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
            
            frame = cv2.imread(img_path)
            if frame is None:
                os.remove(meta_path); os.remove(img_path)
                continue

            print(f"[Retry] 🚀 Retrying {os.path.basename(meta_path)}...")
            success = upload_logic_core(frame, data['frame_count'], data['timestamp'], data['latitude'], data['longitude'])

            if success:
                os.remove(meta_path)
                os.remove(img_path)
                print(f"[Retry] ✅ Retry Success! Removed from queue.")
            else:
                print(f"[Retry] ❌ Retry Failed again. Stopping queue processing.")
                break 

        except Exception as e:
            print(f"[Retry] Error processing queue item: {e}")
            break

# -----------------------------------------------------------------------------------------------
# [Helper] API 전송 코어 로직
# -----------------------------------------------------------------------------------------------
def upload_logic_core(frame, frame_count, timestamp, lat, lon):
    try:
        success, encoded_img = cv2.imencode('.jpg', frame)
        if not success: return False

        filename = f"crack_{int(time.time())}_{frame_count}.jpg"
        files = {'file': (filename, encoded_img.tobytes(), 'image/jpeg')}
        
        # 타임아웃 15초, 서버 상황에 따라 변
        res_img = requests.post(URL_IMAGE_UPLOAD, files=files, timeout=15)

        if res_img.status_code not in [200, 201]:
            print(f"[API] ❌ Image Upload Error: {res_img.status_code}")
            return False

        server_img_path = res_img.json().get("url")
        if not server_img_path: return False
            
        payload = {
            "latitude": lat,
            "longitude": lon,
            "image": server_img_path,
            "detect_time": timestamp
        }

        res_meta = requests.post(URL_METADATA, json=payload, timeout=15)

        if res_meta.status_code in [200, 201]:
            print(f"\r[API] ✅ Upload Success! ({timestamp})", end="")
            return True
        else:
            print(f"[API] ❌ Metadata Error: {res_meta.status_code}")
            return False

    except Exception as e:
        print(f"[API] ❌ Network Exception: {e}")
        return False

# -----------------------------------------------------------------------------------------------
# [Helper] 메인 프로세스 핸들러
# -----------------------------------------------------------------------------------------------
def get_frame_timestamp(file_mtime, current_sec):
    base_time = datetime.fromtimestamp(file_mtime)
    detect_dt = base_time + timedelta(seconds=current_sec)
    return detect_dt.strftime("%Y-%m-%d %H:%M:%S")

def process_upload_sequence(frame, frame_count, timestamp, lat, lon, status_tracker, current_sec):
    system_now = datetime.now().strftime("%H:%M:%S")
    print(f"\n[Debug] 🚀 POST Start: {system_now} | Video Pos: {current_sec:.2f}s")

    success = upload_logic_core(frame, frame_count, timestamp, lat, lon)

    if success:
        print(f"\r[API] ✅ Current Upload Success! (Vid: {current_sec:.1f}s)", end="")
        process_retry_queue() 
    else:
        print(f"\n[API] ⚠️ Failed. Saving to retry queue.")
        save_failed_request(frame, frame_count, timestamp, lat, lon)
        status_tracker['failed'] = True

# -----------------------------------------------------------------------------------------------
# [Core] Decoder Class (NMS)
# -----------------------------------------------------------------------------------------------
class YOLOSegDecoder:
    def __init__(self, input_size=(640, 640), num_classes=1, conf_thres=0.25, iou_thres=0.45):
        self.input_size = input_size
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.strides = [8, 16, 32]
        self.grids = {s: self._make_grid(input_size[0] // s, input_size[1] // s) for s in self.strides}

    def _make_grid(self, nx, ny):
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        return np.stack((xv, yv), 2).reshape((1, -1, 2)).astype(np.float32)

    def _sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def _softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def decode(self, raw_tensors):
        proto = None
        outputs_reg, outputs_cls, outputs_mask_coef = [], [], []
        for name, t in raw_tensors.items():
            if len(t.shape) == 4 and t.shape[0] == 1: t = t[0]
            if len(t.shape) == 2: t = np.expand_dims(t, axis=-1)
            try: h, w, c = t.shape
            except ValueError: continue
            if h == 160 and w == 160 and c == 32: proto = t 
            elif c == 64: outputs_reg.append((h, t))
            elif c == NUM_CLASSES: outputs_cls.append((h, t))
            elif c == 32: outputs_mask_coef.append((h, t))

        if proto is None or not outputs_reg: return [], []
        outputs_reg.sort(key=lambda x: x[0], reverse=True)
        outputs_cls.sort(key=lambda x: x[0], reverse=True)
        outputs_mask_coef.sort(key=lambda x: x[0], reverse=True)
        
        all_boxes, all_scores, all_mask_coefs = [], [], []
        for i, stride in enumerate(self.strides):
            if i >= len(outputs_reg): break
            reg_feat, cls_feat, mask_feat = outputs_reg[i][1], outputs_cls[i][1], outputs_mask_coef[i][1]
            reg_flat = reg_feat.reshape(-1, 64)
            cls_flat = cls_feat.reshape(-1, self.num_classes)
            mask_flat = mask_feat.reshape(-1, 32)
            cls_scores = self._sigmoid(cls_flat)
            mask = np.max(cls_scores, axis=1) > self.conf_thres
            if not np.any(mask): continue
            
            reg_sel = reg_flat[mask]
            cls_sel = cls_scores[mask]
            mask_coef_sel = mask_flat[mask]
            grid_sel = self.grids[stride].reshape(-1, 2)[mask]
            reg_sel = reg_sel.reshape(-1, 4, 16)
            prob = self._softmax(reg_sel, axis=2)
            dist = np.sum(prob * np.arange(16), axis=2)
            cx = grid_sel[:, 0] + (dist[:, 2] - dist[:, 0]) / 2
            cy = grid_sel[:, 1] + (dist[:, 3] - dist[:, 1]) / 2
            w, h = dist[:, 0] + dist[:, 2], dist[:, 1] + dist[:, 3]
            x1, y1 = (cx - w/2)*stride, (cy - h/2)*stride
            x2, y2 = (cx + w/2)*stride, (cy + h/2)*stride
            all_boxes.append(np.stack((x1, y1, x2, y2), axis=1))
            class_ids = np.argmax(cls_sel, axis=1)
            all_scores.append(cls_sel[np.arange(len(cls_sel)), class_ids])
            all_mask_coefs.append(mask_coef_sel)

        if not all_boxes: return [], []
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_mask_coefs = np.concatenate(all_mask_coefs, axis=0)
        indices = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(), self.conf_thres, self.iou_thres)
        
        final_boxes, final_masks = [], []
        if len(indices) > 0:
            indices = indices.flatten()
            for idx in indices:
                box = all_boxes[idx]
                mask_raw = np.dot(proto, all_mask_coefs[idx]) 
                mask_sig = self._sigmoid(mask_raw)
                x1, y1, x2, y2 = (box / 4).astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(160, x2), min(160, y2)
                final_mask = np.zeros((160, 160), dtype=np.float32)
                final_mask[y1:y2, x1:x2] = mask_sig[y1:y2, x1:x2]
                final_mask = (final_mask > 0.5).astype(np.uint8) * 255
                final_boxes.append((box, all_scores[idx]))
                final_masks.append(final_mask)
        return final_boxes, final_masks

# -----------------------------------------------------------------------------------------------
# [Function] 외부 실행 함수
# -----------------------------------------------------------------------------------------------
def run_video_inference(video_path):
    if not os.path.exists(HEF_PATH):
        print(f"[Error] HEF not found: {HEF_PATH}"); return False
    
    try: file_mtime = os.path.getmtime(video_path)
    except: file_mtime = time.time()

    print(f"[GPS] Searching for reference photo...")
    ref_image = find_latest_image_before_video(video_path)
    gps_lat, gps_lon = 0.0, 0.0
    if ref_image:
        lat, lon = get_gps_from_image(ref_image)
        if lat and lon: 
            gps_lat, gps_lon = lat, lon
            print(f"[GPS] ✅ Coordinates Found: {gps_lat:.6f}, {gps_lon:.6f}")
        else:
            print(f"[GPS] ⚠️ Photo found, but no GPS data.")
    else:
        print("[GPS] ⚠️ No reference photo found.")

    upload_status = {'failed': False}
    decoder = YOLOSegDecoder(conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD, num_classes=NUM_CLASSES)
    hef = HEF(HEF_PATH)

    print(f"\n[Hailo] Initializing Device for {os.path.basename(video_path)}...")
    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()
        
        input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        
        input_info = hef.get_input_vstream_infos()[0]
        model_w, model_h = input_info.shape[1], input_info.shape[0]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        FRAME_INTERVAL = int(fps * 5) # 5초 간격, 상황에 따라 조

        print(f"[Hailo] Processing every {FRAME_INTERVAL} frames (5.0s).")
        frame_count = 0

        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if frame_count % FRAME_INTERVAL == 0:
                        input_frame = cv2.resize(frame, (model_w, model_h))
                        input_data = np.expand_dims(input_frame.astype(np.float32), axis=0)

                        infer_results = infer_pipeline.infer(input_data)
                        final_boxes, final_masks = decoder.decode(infer_results)
                        
                        if len(final_boxes) > 0:
                            save_img = input_frame.copy()
                            for i, (box, score) in enumerate(final_boxes):
                                x1, y1, x2, y2 = map(int, box)
                                x1 = max(0, min(x1, 639))
                                y1 = max(0, min(y1, 639))
                                x2 = max(0, min(x2, 639))
                                y2 = max(0, min(y2, 639))
                                #mask = final_masks[i]
                                #mask_resized = cv2.resize(mask, (model_w, model_h), interpolation=cv2.INTER_NEAREST)
                                #colored_mask = np.zeros_like(save_img)
                                #colored_mask[mask_resized > 0] = (0, 255, 0)
                                #save_img = cv2.addWeighted(save_img, 1.0, colored_mask, 0.5, 0)  # 시각화 1. segment 표시
                                cv2.rectangle(save_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 시각화 2. bounding box 표시
                                #label = f"{score:.2f}"
                                #(t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                                #if y1 - t_h - 5 < 0:
                                #    text_y = y1 + t_h + 5
                                #else:
                                #    text_y = y1 - 5
                                #if x1 + t_w > img_w:
                                #    text_x = img_w - t_w - 2
                                #else:
                                #    text_x = x1
                                #cv2.putText(save_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 시각화 3. confidence 표시
                                                        
                            current_sec = frame_count / fps
                            timestamp_str = get_frame_timestamp(file_mtime, current_sec)

                            print(f"\r[Detect] Fr {frame_count} | Sending...", end="")
                            
                            # 전송 함수 호출 (Blocking)
                            #t = Thread(target=process_upload_sequence, args=(save_img, frame_count, timestamp_str, gps_lat, gps_lon, upload_status))
                            #t.start()  # 쓰레딩으로 분산처리, 현재 서버 과부하 문제로 아래 한단계씩 처리 중
                            
                            process_upload_sequence(save_img, frame_count, timestamp_str, gps_lat, gps_lon, upload_status, current_sec)
                        else:
                            print(f"\r[Processing] Frame {frame_count}...", end="")
                    
                    frame_count += 1

        cap.release()
        
        # 영상 처리가 끝나면 남은 큐 처리 시도
        process_retry_queue()

        if upload_status['failed']:
            print(f"\n[Core] ⚠️ Upload errors occurred (Some items saved to queue).")
            return True 
        else:
            print(f"\n[Hailo] ✅ Finished successfully.")

            return True
