# AgriVision-AI 🌾🤖

> **Intelligent Agricultural Image Classification & Decision Support Platform**  
> *Nền tảng Thị giác Máy tính Thông minh trong Phân loại và Giám sát Nông nghiệp*

---

## 📌 Giới thiệu Tổng quan (Overview)

**AgriVision-AI** là giải pháp phần mềm ứng dụng Trí tuệ Nhân tạo (Deep Learning) toàn diện phục vụ lĩnh vực nông nghiệp hiện đại. Hệ thống được thiết kế theo kiến trúc module hóa từ khâu nghiên cứu, huấn luyện mô hình học sâu cho đến khâu triển khai thành sản phẩm phần mềm thực tế với API serving và giao diện trực quan cho người dùng cuối.

---

## 🏛 Kiến trúc Hệ thống (System Architecture)

Dự án được đóng gói trọn gói bao gồm 3 phân hệ chính:

```text
AgriVision-AI/
├── project1_api/        # [Backend] RESTful API phục vụ suy luận mô hình (Model Serving)
│   ├── app/
│   │   ├── main.py          # FastAPI application entrypoint
│   │   ├── model_loader.py  # Quản lý nạp checkpoint và tối ưu hóa bộ nhớ
│   │   ├── predict.py       # Logic tiền xử lý và inference
│   │   └── schemas.py       # Pydantic schemas (Request / Response validation)
│   └── requirements.txt
│
├── project2_ui/         # [Frontend] Giao diện ứng dụng người dùng tương tác
│   ├── app.py               # Ứng dụng Web trực quan (Dashboard phân tích ảnh nông nghiệp)
│   └── requirements.txt
│
├── project3_model/      # [AI Core] Pipeline nghiên cứu & huấn luyện Deep Learning
│   ├── AI-for-Agriculture-2026/  # Pipeline PyTorch, late-fusion, experiment tracking
│   ├── Log thực nghiệm/          # Bảng theo dõi và đánh giá metrics
│   └── ảnh confusion matrix/     # Trực quan hóa kết quả đánh giá (Evaluation)
│
├── img/                 # Tài nguyên hình ảnh, branding & logo
└── README.md            # Tài liệu tổng quan sản phẩm
```

---

## 🚀 Các tính năng chính (Key Features)

- **AI Core mạnh mẽ:** Tích hợp các kiến trúc mạng nơ-ron tiên tiến (ConvNeXt, ResNet, EfficientNet) hỗ trợ cả dữ liệu ảnh quang học (RGB) lẫn đa phổ (Multimodal).
- **Backend API chuẩn công nghiệp:** Xây dựng trên nền tảng FastAPI với tài liệu Swagger tự động, độ trễ thấp, dễ dàng mở rộng và tích hợp vào các hệ thống IoT/Smart Farming.
- **Giao diện Web tương tác:** Cho phép người dùng tải ảnh lên, nhận kết quả chẩn đoán/phân loại theo thời gian thực kèm độ tin cậy (confidence score) và trực quan hóa phân tích.

---

## ⚙️ Hướng dẫn Khởi chạy (Quick Start)

### 1. Khởi chạy Backend API (`project1_api`)
```bash
cd project1_api
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
*Tài liệu API Swagger UI:* Truy cập `http://localhost:8000/docs`

### 2. Khởi chạy Giao diện Người dùng (`project2_ui`)
```bash
cd project2_ui
pip install -r requirements.txt
streamlit run app.py
```
*Giao diện Web:* Mở trình duyệt tại `http://localhost:8501`

---

## 📄 Bản quyền & Tác giả (License & Credits)
- **Tên sản phẩm:** AgriVision-AI
- **Đơn vị phát triển:** IT SGU - Khoa Công nghệ Thông tin, Trường Đại học Sài Gòn
