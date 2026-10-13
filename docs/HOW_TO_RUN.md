# How to Run: Real-Time Industrial Defect Detection System

## 1. What you need

- Windows with PowerShell, Python 3.10 or newer, and Git
- An NVIDIA GPU is recommended for training (Google Colab's free T4 works). The data scripts, inference scripts and the API all run on CPU.

## 2. Setup

    git clone https://github.com/deepthi857/project_2.git
    cd project_2
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

If PowerShell refuses to run the activate script:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

On an older CPU without AVX support (a Polars warning appears), run this before any Python script:

    $env:POLARS_SKIP_CPU_CHECK=1

## 3. Dataset

The NEU-DET dataset is not stored in the repository. Place it like this:

    data/raw/NEU-DET/train/annotations/        (VOC XML files)
    data/raw/NEU-DET/train/images/<class>/     (jpg files)
    data/raw/NEU-DET/validation/annotations/
    data/raw/NEU-DET/validation/images/<class>/

The six classes are crazing, inclusion, patches, pitted_surface, rolled-in_scale and scratches.

## 4. Data pipeline (run from the project root, in this order)

1. Convert annotations from VOC XML to YOLO format (the file name has a historical typo, use it as is):

        python src\data\conert_annotations.py

2. Generate augmented images (rotation, noise, lighting), 2 per original image:

        python src\data\augment.py

3. Build the flat training folders:

        python src\data\prepare_training_data.py

4. YOLO expects labels next to the images, so move the combined labels into place:

        del data\labels\train\*.txt
        move data\labels\train_combined\*.txt data\labels\train\

5. Check the result:

        python src\data\check_data_integrity.py

   Expected: 4318 training images with 4317 labels (crazing_240 has no label), 360 validation images.

6. Optional visual check of the bounding boxes (writes verify_output.jpg):

        python src\data\visualize.py

## 5. Training

    python src\train\train_model.py

Training on a CPU-only laptop took about 1 hour 19 minutes per epoch, so use a GPU. On a Colab T4, 50 epochs took about 0.9 hours and reached mAP50 = 0.727 and mAP50-95 = 0.401.

The trained weights are written to runs\detect\train\weights\best.pt. Copy that file to models\best.pt. The models folder is git-ignored, so share the file separately.

Evaluation tools (the first one needs models\best.pt):

    python src\data\class_performance_report.py
    python src\data\metrics.py

## 6. Export and edge inference

    python src\edge\export_onnx.py
    python src\edge\check_runtime.py
    python src\edge\video_inference_onnx.py

Before exporting your own model, set model_path inside export_onnx.py to models\best.pt. TensorRT export (export_tensorrt.py) needs an NVIDIA GPU with CUDA. On a CPU-only machine it fails with a clear message, and ONNX Runtime is used instead.

## 7. API

    uvicorn src.api.main:app --reload

Open these in a browser:

- http://127.0.0.1:8000/docs for interactive testing
- http://127.0.0.1:8000/health for service status and the model in use
- http://127.0.0.1:8000/metrics for Prometheus metrics

POST /detect takes a multipart upload with the field name `file` and returns each defect's class, confidence and bounding box. A file that is not an image or is empty returns 400, and a file over 10 MB returns 413.

The API loads models\best.pt if it exists. Otherwise it falls back to the generic yolov8n.pt, which finds no defects.

Run the input-handling tests:

    python src\api\test_api_inputs.py

## 8. Monitoring

| Metric | Meaning |
|---|---|
| model_inference_latency_seconds | Time the model takes per image |
| defects_detected_total{defect_class} | Defects found, per class |
| images_processed_total | Images processed successfully |
| defective_images_total | Images with at least one defect |
| invalid_requests_total{reason} | Rejected uploads, by reason |
| service_uptime_seconds | Seconds since the API started |
| camera_up | 1 if the camera sent a frame in the last 10 seconds |
| plc_messages_sent_total, plc_send_errors_total | PLC messages sent and failed |

Standard http_* request metrics are also exposed.

## 9. PLC broadcast and camera uptime

Every /detect call sends a UDP JSON message to PLC_HOST:PLC_PORT (default 127.0.0.1:9000) with the action (PASS or REJECT), the defect count, the image size and each defect's class, confidence and bounding box. Change the target with the environment variables PLC_HOST, PLC_PORT and PLC_ENABLED.

Demo in three terminals, each with the venv active:

    python src\api\mock_plc.py
    uvicorn src.api.main:app --reload
    python src\edge\camera_client.py

The camera client sends each frame to /detect and calls /camera/heartbeat. If no frame arrives for 10 seconds, camera_up drops to 0. Check it at http://127.0.0.1:8000/camera/status. Set CAMERA_SOURCE=0 to use a webcam, or point it at a video file or an image folder.

## 10. Docker and the monitoring stack

    docker compose up --build

- API: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (login admin / admin; the Prometheus datasource is already provisioned)

This needs Docker Desktop. It has not yet been verified on the development laptop, where Docker was not installed. The container uses the generic model unless models\best.pt was present when the image was built.

Suggested Grafana panels (query for each):

| Panel | Query |
|---|---|
| Inference latency p95 | histogram_quantile(0.95, rate(model_inference_latency_seconds_bucket[1m])) |
| Throughput | rate(images_processed_total[1m]) |
| Defect rate | rate(defective_images_total[5m]) / rate(images_processed_total[5m]) |
| Defects per class (last hour) | sum by (defect_class) (increase(defects_detected_total[1h])) |
| Camera up | camera_up |
| Camera uptime, last hour | avg_over_time(camera_up[1h]) |
| PLC send errors | rate(plc_send_errors_total[5m]) |

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| Script exits with code -1073741795 (Illegal Instruction) | The CPU lacks AVX instructions. Train on Google Colab instead. |
| ModuleNotFoundError | The virtual environment is not active. Run `.venv\Scripts\activate`. |
| "running scripts is disabled" | Run the Set-ExecutionPolicy command from section 2. |
| Directory not found: data/raw/NEU-DET | The dataset is not extracted into data\raw yet. |
| API returns no detections | It is using yolov8n.pt. Put the trained model at models\best.pt. |
