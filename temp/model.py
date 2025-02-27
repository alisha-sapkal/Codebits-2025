from PIL import Image, ImageDraw, ImageFont
import pytesseract
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from typing import Dict, Any
import requests
import json

class MPSCQuestionAnswerer:
    def __init__(self):
        # Update this path to match your Tesseract installation path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Initialize BERT model and tokenizer
        self.model_name = "deepset/roberta-base-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        
        # Load MPSC context database
        self.context_database = self.load_context_database()

    def load_context_database(self) -> Dict[str, str]:
        """
        Load or initialize the knowledge base for MPSC
        Returns:
            dict: Dictionary containing topic-wise information
        """
        return {
            "history": """
                Maharashtra's rich history includes the Marathas, Shivaji Maharaj, and Peshwas.
                The Maratha Empire was established by Chhatrapati Shivaji Maharaj in 1674.
                Important historical periods include:
                - Ancient Period: Satavahanas, Vakatakas, Chalukyas
                - Medieval Period: Yadavas, Bahmani Kingdom
                - Maratha Period: Shivaji's administration, Peshwa rule
                - Modern Period: Social reformers like Mahatma Phule, Dr. Ambedkar
                The Samyukta Maharashtra Movement led to the formation of Maharashtra state in 1960.
            """,
            
            "geography": """
                Maharashtra is India's third-largest state by area.
                Key geographical features:
                - Western Ghats (Sahyadri Range)
                - Konkan Coast
                - Deccan Plateau
                - Major rivers: Godavari, Krishna, Tapi, Narmada
                - Climate zones: Tropical monsoon
                - Forest types: Semi-evergreen, deciduous
                - Important wildlife sanctuaries: Tadoba, Melghat
                Major cities: Mumbai, Pune, Nagpur, Nashik
            """,
            
            "polity": """
                Maharashtra state government structure:
                - Legislative Assembly (Vidhan Sabha): 288 members
                - Legislative Council (Vidhan Parishad): 78 members
                - Governor: Constitutional head
                - Chief Minister: Head of government
                Local government:
                - Municipal Corporations
                - Zilla Parishads
                - Panchayati Raj system
                Important administrative divisions: 6 divisions, 36 districts
            """,
            
            "economy": """
                Maharashtra's economy:
                - Largest state economy in India
                - Major sectors: IT, manufacturing, agriculture
                - Important crops: Cotton, sugarcane, jowar
                - Industrial hubs: MIDC areas
                - Mumbai: Financial capital of India
                - Major ports: JNPT, Mumbai Port
                Agricultural patterns and cropping seasons
            """,
            
            "current_affairs": """
                Recent developments in Maharashtra:
                - Infrastructure projects
                - Government schemes
                - Social welfare programs
                - Environmental initiatives
                - Industrial policies
                Updates on major state projects
            """,
            
            "culture": """
                Maharashtra's cultural heritage:
                - Languages: Marathi, regional dialects
                - Folk arts: Lavani, Povada
                - Festivals: Ganesh Chaturthi, Gudi Padwa
                - Literature: Sant tradition, modern literature
                - Traditional arts: Warli, Chitrakathi
                - Classical music: Natya Sangeet
            """,
            
            "general": """
                Maharashtra is one of India's largest and most significant states.
                Key aspects:
                - Capital: Mumbai (financial capital of India)
                - Founded: 1 May 1960
                - Language: Marathi
                - Notable features: Rich history, diverse culture, strong economy
                - Important figures: Shivaji Maharaj, Babasaheb Ambedkar, Mahatma Phule
                - Major sectors: IT, agriculture, manufacturing
                - Cultural highlights: Festivals, arts, literature
                - Geographic features: Western Ghats, Deccan Plateau, Konkan Coast
            """
        }

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            # Try to check tesseract version to verify installation
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                return "Error: Tesseract is not installed. Please install Tesseract OCR first."
            
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                return "Error: No text detected in the image"
            
            return text.strip()
        except Exception as e:
            if "tesseract is not installed" in str(e):
                return "Error: Please install Tesseract OCR and set the correct path"
            return f"Error processing image: {str(e)}"

    def classify_question_topic(self, question: str) -> str:
        """
        Classify the question to identify relevant context
        Args:
            question (str): The question text
        Returns:
            str: Topic category
        """
        # Simple keyword-based classification (can be enhanced with ML)
        keywords = {
            "history": ["history", "ancient", "empire", "dynasty", "ruler", "kingdom", "shivaji", "maharaj", "maratha"],
            "geography": ["geography", "climate", "river", "mountain", "plateau"],
            "polity": ["constitution", "parliament", "democracy", "government", "law"]
        }
        
        question = question.lower()
        for topic, topic_keywords in keywords.items():
            if any(keyword in question for keyword in topic_keywords):
                return topic
        return "general"

    def get_answer(self, question: str, context: str) -> Dict[str, Any]:
        """
        Get answer for a question using BERT
        Args:
            question (str): The question text
            context (str): The context for answering
        Returns:
            dict: Answer and confidence score
        """
        try:
            inputs = self.tokenizer.encode_plus(
                question, 
                context, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )

            outputs = self.model(**inputs)
            
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits)
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            answer = self.tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end+1])
            
            confidence = float(torch.max(outputs.start_logits)) + float(torch.max(outputs.end_logits))
            confidence = min(1.0, max(0.0, confidence / 10.0))  # Normalize confidence

            return {
                "answer": answer,
                "confidence": confidence,
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating answer: {str(e)}"
            }

    def process_question_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image containing a question and return the answer
        Args:
            image_path (str): Path to the image file
        Returns:
            dict: Dictionary containing the answer and metadata
        """
        if not os.path.exists(image_path):
            return {"status": "error", "message": "Image file not found"}

        # Extract text from image
        question_text = self.extract_text_from_image(image_path)
        if question_text.startswith("Error"):
            return {"status": "error", "message": question_text}

        # Classify question topic
        topic = self.classify_question_topic(question_text)
        context = self.context_database.get(topic, self.context_database["general"])

        # Get answer
        result = self.get_answer(question_text, context)
        if result["status"] == "success":
            return {
                "status": "success",
                "question": question_text,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "topic": topic
            }
        return result

# Add this function to create a test image
def create_test_image(text: str, filename: str = "test_question.png"):
    """Create a test image with the given text"""
    # Create a new image with white background
    img = Image.new('RGB', (800, 200), color='white')
    d = ImageDraw.Draw(img)
    
    # Use default font
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    # Add text to image
    d.text((20, 50), text, font=font, fill='black')
    
    # Save the image
    img.save(filename)
    return filename

# Modify the main section for better error handling
if __name__ == "__main__":
    qa_system = MPSCQuestionAnswerer()
    
    print("🔍 Starting analysis...")
    
    # Create a test image with a question
    test_question = "Who was Chhatrapati Shivaji Maharaj and what was his contribution to Maharashtra?"
    try:
        image_path = create_test_image(test_question)
        print("✅ Test image created successfully")
    except Exception as e:
        print(f"❌ Error creating test image: {str(e)}")
        exit(1)
    
    print("\n1. Reading image...")
    question_text = qa_system.extract_text_from_image(image_path)
    
    if question_text.startswith("Error"):
        print(f"❌ {question_text}")
        print("\nPlease make sure Tesseract OCR is installed:")
        print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("- Linux: sudo apt-get install tesseract-ocr")
        print("- Mac: brew install tesseract")
        exit(1)
    
    print(f"2. Detected text: {question_text}")
    print("3. Processing question...")
    
    result = qa_system.process_question_image(image_path)
    
    if result["status"] == "success":
        print("\n=== Results ===")
        print("📝 Question:", result["question"])
        print("💡 Answer:", result["answer"])
        print("✨ Confidence: {:.2f}%".format(result["confidence"]*100))
        print("📚 Topic:", result["topic"])
        print("===============")
    else:
        print("❌ Error:", result["message"])
