import pdfreader
from PyPDF2 import PdfReader
from pdfreader import SimplePDFViewer


def batch_translate(text_list, num_elements=None, model=None, tokenizer=None):
    translated_text = []
    
    # Determine how many elements to process
    if num_elements is not None:
        text_list = text_list[:num_elements]  # Limit to the specified number of elements

    for text in text_list:
        if not text.strip():  # Skip empty strings
            continue  
        model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_tokens = model.generate(**model_inputs, max_length=4000)
        translated_chunk = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_text.append(translated_chunk[0])
    
    # Return the list of translated chunks
    return translated_text

def extract_full(pdf_path):
    with open(pdf_path, "rb") as file:
        viewer = SimplePDFViewer(file)
        viewer.render()
        text = ""
        for page in viewer:
            text += "".join(page.strings)
    return text


def extract_batch(pdf_path, max_pages=None):
    reader = PdfReader(pdf_path)
    text = ""
    
    # Determine the number of pages to process
    total_pages = len(reader.pages)
    pages_to_process = total_pages if max_pages is None else min(max_pages, total_pages)
    
    for page_number in range(pages_to_process):
        page = reader.pages[page_number]
        text += page.extract_text() + "\n"  # Add a newline for separation between pages
    
    return text







