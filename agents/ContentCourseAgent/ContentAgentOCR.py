# ocr_loader.py
import os
import base64
from mistralai import Mistral

class MistralOCRLoader:
    def __init__(self, api_key, output_image_dir="course_images"):
        self.client = Mistral(api_key=api_key)
        self.output_image_dir = output_image_dir
        os.makedirs(output_image_dir, exist_ok=True)

    def process_file(self, file_path):
        """
        Отправляет файл в Mistral OCR, сохраняет картинки и возвращает
        Markdown текст и словарь с описаниями картинок (пока пустой).
        """
        filename = os.path.basename(file_path)
        print(f"OCR обработка: {filename}...")
        
        # 1. Загрузка файла
        with open(file_path, "rb") as f:
            uploaded_file = self.client.files.upload(
                file={"file_name": filename, "content": f},
                purpose="ocr"
            )
        
        # 2. Получение URL и OCR
        signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url",
                       "document_url": signed_url.url},
            include_image_base64=True
        )
        
        # 3. Сборка результатов
        full_markdown = ""
        saved_images = {} # { "img-id": "path/to/file.jpg" }
        
        for page in ocr_response.pages:
            full_markdown += page.markdown + "\n\n"
            
            # Обработка картинок на странице
            for img in page.images:
                img_id = img.id # id внутри markdown (напр. "img-0.jpeg")
                b64_data = img.image_base64
                
                # Очистка base64
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                
                # Сохранение на диск
                img_filename = f"{os.path.splitext(filename)[0]}_{img_id}"
                img_path = os.path.join(self.output_image_dir, img_filename)
                
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                
                saved_images[img_id] = {
                    "path": img_path,
                    "base64": b64_data # Сохраняем временно для Vision модели
                }
        
        # Удаляем файл из облака Mistral
        self.client.files.delete(file_id=uploaded_file.id)
        
        return full_markdown, saved_images
