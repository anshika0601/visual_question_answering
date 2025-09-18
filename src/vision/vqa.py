# src/vision/vqa.py
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch

class VQAEngine:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading VQA model on {self.device}...")
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base"
        ).to(self.device)
        
        print("VQA model loaded successfully")
        
    def answer_question(self, image, question):
        """Answer question about image"""
        try:
            # Prepare inputs
            inputs = self.processor(
                image, question, return_tensors="pt"
            ).to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
                
            # Decode answer
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "success": True,
                "answer": answer,
                "question": question
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}