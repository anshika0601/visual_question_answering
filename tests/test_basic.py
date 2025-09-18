# tests/test_basic.py
from visual_question_answering.src.init import OmniAssist

def test_basic_functionality():
    """Test basic image loading and processing"""
    app = OmniAssist()
    
    # Test with example image
    result = app.process_image(
        "visual_question_answering/durgamaa.jpg",
        "What is in this image?"
    )
    
    print("Test Results:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Original size: {result['original_size']}")
        print(f"Processed size: {result['processed_size']}")
        if result['vqa_result']:
            print(f"Question: {result['vqa_result']['question']}")
            print(f"Answer: {result['vqa_result']['answer']}")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    test_basic_functionality()