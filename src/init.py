# src/main.py
from .utils.img_processor import ImageProcessor
from .vision.vqa import VQAEngine

class OmniAssist:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.vqa=VQAEngine()
        print("OmniAssist initialized")
        
    def process_image(self, image_path, question=None):
        """Main processing method"""
        try:
            # Load and validate image
            image = self.image_processor.load_image(image_path)
            is_valid, message = self.image_processor.validate_image(image)

            if not is_valid:
                return {"success": False, "error": message}

            # Preprocess for models
            processed_image = self.image_processor.preprocess_for_model(image)

            # If question provided, answer it
            vqa_result = None
            if question:
                vqa_result = self.vqa.answer_question(processed_image, question)

            return {
                "success": True,
                "original_size": image.size,
                "processed_size": processed_image.size,
                "vqa_result": vqa_result

            }

        except Exception as e:
            return {"success": False, "error": str(e)}

