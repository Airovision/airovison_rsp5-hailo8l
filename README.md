# airovison_rsp5-hailo

# Airovision: Edge AI Crack Detection System on Raspberry Pi 5

**Airovision**은 Raspberry Pi 5와 Hailo-8L AI 가속기를 활용하여 건물 외벽의 균열을 실시간으로 탐지하는 엣지 컴퓨팅 시스템입니다. 드론이 촬영한 영상을 SD카드 또는 네트워크 폴더를 통해 자동으로 수집하고, AI 추론을 수행한 뒤 결과를 클라우드 서버로 전송합니다.

## 📌 주요 기능 (Key Features)

1.  **Dual Mode Monitoring (하이브리드 감시)**:
    * **Mode 1 (Stream)**: 실시간 통신 폴더(`drone_stream_data`)에 영상이 수신되면 최우선으로 분석합니다.
    * **Mode 2 (SD Card)**: SD카드가 마운트되면 자동으로 감지하여 내부의 영상을 분석합니다.
2.  **High-Performance Inference**:
    * Hailo-8L NPU를 사용하여 YOLOv11 Segmentation 모델을 고속으로 구동합니다.
    * Python NumPy 기반의 Custom Post-Processing(NMS, Decoding) 엔진을 탑재하여 Raw Tensor를 직접 해석합니다.
3.  **Fault-Tolerant Transmission (무중단 전송)**:
    * 네트워크 연결이 불안정하여 서버 전송에 실패할 경우, 데이터를 로컬 큐(`failed_uploads`)에 저장합니다.
    * 다음 전송 성공 시 큐에 쌓인 데이터를 자동으로 재전송하여 데이터 유실을 방지합니다.
4.  **GPS Synchronization**:
    * 영상 파일에는 없는 GPS 정보를 보완하기 위해, 촬영 시점 직전의 사진 파일(EXIF)을 찾아 위치 정보를 동기화합니다.

---

## 🛠️ 하드웨어 및 소프트웨어 요구사항

* **Hardware**: Raspberry Pi 5, Hailo-8L AI Accelerator (M.2 HAT)
* **OS**: Raspberry Pi OS (64-bit)
* **Environment**: Hailo TAPPAS & HailoRT installed

---

## 🚀 설치 및 설정 가이드 (Installation)

### 0. hef 파일 다운로드

다음 림크에서 학습한 yolov11-segment hef 파일을 다운받을 수 있습니다.

[hef 파일 다운로드 링크](https://drive.google.com/file/d/1laIM1E0uGLtLy8anqVN-O_QX_yhETDit/view?usp=sharing)

### 1. Hailo-8L 환경 세팅
이 프로젝트는 Hailo 공식 예제 환경을 기반으로 합니다. 먼저 아래 공식 가이드를 따라 기본 환경을 구축하세요.

1.  [Hailo RPi5 Examples Repository](https://github.com/hailo-ai/hailo-rpi5-examples)를 클론합니다.
2.  `setup_env.sh` 등을 통해 가상환경(`venv_hailo_rpi_examples`)을 생성하고 Hailo Driver 및 Software를 설치합니다.

### 2. 프로젝트 코드 설치
생성된 Hailo 가상환경 폴더 내(혹은 작업 폴더)에 다음 파일들을 위치시킵니다.
* `main_monitor.py`: 메인 실행 파일 (모니터링)
* `inference_core.py`: AI 추론 및 통신 핵심 모듈
* `gps_helper.py`: GPS 데이터 추출 헬퍼
* `processed_history.json`: (자동생성됨) 중복 처리 방지 기록

### 3. 의존성 패키지 설치
가상환경을 활성화한 상태에서 필요한 Python 패키지를 설치합니다.
(제공된 `requirements.txt`를 사용하거나, 필수 패키지만 직접 설치할 수 있습니다.)

```bash
# 가상환경 활성화 (예시)
source venv_hailo_rpi_examples/bin/activate

# 필수 패키지 설치
pip install requests psutil opencv-python numpy Pillow
```

### ⚙️ 설정 (Configuration)
실행 전, 본인의 환경에 맞게 코드를 수정해야 합니다.

1. 서버 주소 설정 (inference_core.py)
#### 1. 서버 주소 설정 (`inference_core.py`)
협업하는 백엔드 서버의 IP 주소로 변경하세요.
```python
# inference_core.py 상단
SERVER_BASE_URL  = ""  # <-- 본인의 서버 IP로 수정
```

Python

#### 2. 모델 파일 경로 설정 (`inference_core.py`)
사용할 HEF 모델 파일의 절대 경로를 지정하세요.
```python
# inference_core.py 상단
SERVER_BASE_URL  = ""  # <-- 본인의 서버 IP로 수정
2. 모델 파일 경로 설정 (inference_core.py)
사용할 HEF 모델 파일의 절대 경로를 지정하세요.
HEF_PATH = "" # <-- HEF 파일 경로로 수정
```

Python

# inference_core.py 상단
HEF_PATH = ""
3. 입력 경로 설정 (main_monitor.py)
#### 3. 입력 경로 설정 (`main_monitor.py`)
SD카드가 마운트되는 경로와 드론 영상이 저장되는 폴더명을 확인하세요.

Python

```python
# main_monitor.py 상단
SD_MOUNT_ROOT = "/media/"  # SD카드 마운트 위치
SD_TARGET_FOLDER = "DCIM/DJI_001"           # 영상이 저장된 내부 폴더
SD_MOUNT_ROOT = "/media/"      # SD카드 마운트 위치
SD_TARGET_FOLDER = "DCIM/DJI_001"              # 영상이 저장된 내부 폴더
STREAM_ROOT = "" # 실시간 전송 폴더
▶️ 실행 방법 (Usage)
```

### ▶️ 실행 방법 (Usage)
모든 설정이 완료되면, 터미널에서 모니터링 스크립트를 실행합니다.

Bash

```bash
# 가상환경 활성화
source setup_env.sh  (또는 activate 경로)

# 모니터링 시작
python main_monitor.py
```

프로그램이 실행되면 다음과 같이 대기 모드에 들어갑니다.

Plaintext

```text
=================================================
   AIROVISION MONITORING SYSTEM (Dual Mode)   
   [Mode 1] Real-time Stream Folder Monitor      
   [Mode 2] SD Card Auto-Mount Monitor           
=================================================
[System] Waiting for Connection (Stream/SD)...
```

이 상태에서 USB/SD카드를 꽂거나, 지정된 통신 폴더에 영상 파일을 넣으면 자동으로 감지하여 분석이 시작됩니다.

### 🧠 시스템 작동 프로세스
1. 모니터링 및 감지 (main_monitor.py)
우선순위 감지: STREAM_ROOT 폴더를 먼저 확인하고, 없으면 SD_MOUNT_ROOT를 확인합니다.

중복 방지: processed_history.json 파일을 읽어, 이미 분석이 완료된 영상은 건너뛰고 새로운 영상만 처리합니다.

2. AI 추론 및 후처리 (inference_core.py)
전처리: 입력 영상을 Hailo 모델 입력 크기(640x640)로 리사이즈합니다.

주기적 실행: 영상의 모든 프레임을 분석하지 않고, 효율성을 위해 **설정된 주기(기본 3초)**마다 한 번씩 추론합니다.

Custom Decoding: YOLOv11의 Raw Output을 NumPy 연산으로 직접 디코딩하여 Bounding Box와 Class Score를 추출합니다.

3. 데이터 전송 (inference_core.py - Threaded)
이미지 업로드: 균열이 감지된 프레임(BBox 그려짐)을 /upload-img API로 전송합니다.

메타데이터 전송: 서버로부터 이미지 URL을 받으면, GPS 좌표/시간정보와 함께 /defect-info API로 2차 전송합니다.

재전송 큐(Retry Queue):

네트워크 오류로 전송 실패 시: `failed_uploads/` 폴더에 이미지와 JSON 데이터를 저장합니다.

다음 프레임 전송 성공 시: 네트워크가 복구된 것으로 판단하여, 큐에 쌓인 데이터를 순차적으로 재전송합니다.

### 📂 파일 구조 설명
```text
.
├── main_monitor.py        # [Entry Point] USB/폴더 감시 및 전체 프로세스 제어
├── inference_core.py      # [Core] AI 모델 로드, 추론, 후처리, 서버 통신 로직
├── gps_helper.py          # [Helper] 영상 매칭 이미지 찾기 및 EXIF GPS 추출
├── processed_history.json # 처리된 파일 목록 (자동 생성)
├── failed_uploads/        # 전송 실패한 데이터 임시 저장소 (자동 생성)
└── requirements.txt       # 패키지 의존성 목록
