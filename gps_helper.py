import os
from PIL import Image, ExifTags
from datetime import datetime

def dms_to_dd(dms_tuple, ref):
    """
    (도, 분, 초) 튜플을 십진수(Decimal Degree) 좌표로 변환
    """
    degrees = dms_tuple[0]
    minutes = dms_tuple[1]
    seconds = dms_tuple[2]
    
    dd = degrees + (minutes / 60.0) + (seconds / 3600.0)
    
    # 남위(S)거나 서경(W)이면 음수로 변환
    if ref in ['S', 'W']:
        dd = -dd
        
    return dd

def get_gps_from_image(image_path):
    """
    이미지 파일에서 GPS 정보(위도, 경도)를 추출
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if not exif_data:
            return None, None

        # GPSInfo 태그 ID 찾기
        gps_info = None
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_info = value
                break
        
        if not gps_info:
            return None, None

        # GPSInfo 태그 ID: 1(LatRef), 2(Lat), 3(LonRef), 4(Lon)
        lat_dms = gps_info.get(2)
        lat_ref = gps_info.get(1)
        lon_dms = gps_info.get(4)
        lon_ref = gps_info.get(3)

        if lat_dms and lat_ref and lon_dms and lon_ref:
            lat = dms_to_dd(lat_dms, lat_ref)
            lon = dms_to_dd(lon_dms, lon_ref)
            return lat, lon
            
    except Exception as e:
        print(f"[GPS] Error reading EXIF: {e}")
        
    return None, None

def find_latest_image_before_video(video_path):
    """
    동영상 파일이 있는 폴더에서, 동영상 수정 시간(mtime)보다 
    전에 생성된 파일 중 가장 최신 JPG 이미지
    """
    directory = os.path.dirname(video_path)
    if not directory: directory = "."
        
    try:
        video_mtime = os.path.getmtime(video_path)
        
        candidates = []
        for f in os.listdir(directory):
            if f.lower().endswith(('.jpg', '.jpeg')):
                full_path = os.path.join(directory, f)
                img_mtime = os.path.getmtime(full_path)
                
                # 동영상보다 이전에 찍힌 사진만 후보로 등록
                if img_mtime < video_mtime:
                    candidates.append((img_mtime, full_path))
        
        # 시간이 최신인 순서대로 정렬 (내림차순)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        if len(candidates) > 0:
            best_match = candidates[0][1]
            print(f"[GPS] Reference Image Found: {os.path.basename(best_match)}")
            return best_match
            
    except Exception as e:
        print(f"[GPS] Error searching images: {e}")
        
    return None