from PyPDF2 import PdfReader

def extract_pdf_content(pdf_path, max_pages=None):
    reader = PdfReader(pdf_path)
    text = ""
    
    # Determine the number of pages to process
    total_pages = len(reader.pages)
    pages_to_process = total_pages if max_pages is None else min(max_pages, total_pages)
    
    for page_number in range(pages_to_process):
        page = reader.pages[page_number]
        text += page.extract_text() + "\n"  # Add a newline for separation between pages
    
    return text



def batch_translate(text, chunk_size=5000, model):
    translated_text = ""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        model_inputs = tokenizer(chunk, return_tensors="pt")
        generated_tokens = model.generate(**model_inputs, max_length=4000)
        translated_chunk = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_text += " " + translated_chunk[0]
    return translated_text.strip()