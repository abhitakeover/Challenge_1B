import pymupdf
import json
import re
import os
import numpy as np
import datetime
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

# Configure NLTK to use the data directory we created
nltk.data.path.append("/usr/share/nltk_data")

# Ensure required NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    # If resources are missing, try to download them
    nltk.download('punkt', download_dir='/usr/share/nltk_data', quiet=True)
    nltk.download('stopwords', download_dir='/usr/share/nltk_data', quiet=True)
    nltk.data.path.append('/usr/share/nltk_data')

# Define the base input path
BASE_INPUT_PATH = "/app/input"

def extract_lines(doc):
    """Extract text lines with properties, grouping multi-line headings"""
    lines = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                line_text = ""
                max_size = 0
                bboxes = []
                for span in line["spans"]:
                    line_text += span["text"]
                    if span["size"] > max_size:
                        max_size = span["size"]
                    bboxes.append(span["bbox"])

                if not line_text.strip():
                    continue

                x0 = min(b[0] for b in bboxes)
                y0 = min(b[1] for b in bboxes)
                x1 = max(b[2] for b in bboxes)
                y1 = max(b[3] for b in bboxes)

                lines.append({
                    "text": line_text,
                    "size": max_size,
                    "bbox": (x0, y0, x1, y1),
                    "page": page_num
                })
    return lines

def is_header_footer(text, page_num, total_pages):
    """Identify header/footer content based on common patterns"""
    patterns = [
        r"Copyright", r"©", r"Page \d+ of \d+", r"Version \d+",
        r"ISTQB", r"Qualifications Board", r"International Software Testing",
        r"May 31, 2014", r"Overview.*Agile Tester"
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

def clean_title(title):
    """Clean and normalize the title text"""
    title = re.sub(r"\s+", " ", title)
    title = title.replace("  ", " ").strip()
    title = re.sub(r"Version\s*\d+\.\d+", "", title)
    title = re.sub(r"\d{4}-\d{4}", "", title)
    return title

def process_pdf(pdf_path):
    """Process PDF to extract title and outline with proper page numbering"""
    try:
        doc = pymupdf.open(pdf_path)
        lines = extract_lines(doc)
        total_pages = len(doc)

        # Extract title from first page
        title = ""
        first_page_lines = [line for line in lines if line["page"] == 0]
        if first_page_lines:
            first_page_height = doc[0].rect.height
            top_section = [line for line in first_page_lines if line["bbox"][1] < first_page_height / 2]

            if top_section:
                max_size = max(line["size"] for line in top_section)
                candidate_titles = [line for line in top_section if line["size"] >= max_size * 0.9]
                if candidate_titles:
                    title = " ".join(line["text"].strip() for line in candidate_titles).strip()
                    title = clean_title(title)

        headings = []
        for page_index in range(0, len(doc)):
            page = doc[page_index]
            page_lines = [line for line in lines if line["page"] == page_index]
            page_height = page.rect.height

            page_headings = []

            for line in page_lines:
                y0 = line["bbox"][1]

                # Skip header/footer areas
                if (y0 < page_height * 0.1) or (y0 > page_height * 0.9):
                    continue

                # Skip header/footer content
                if is_header_footer(line["text"], page_index + 1, total_pages):
                    continue

                # Skip title repetitions
                if title and title.strip().lower() == line["text"].strip().lower():
                    continue

                font_size = line["size"]
                level = None
                if font_size > 14:
                    level = "H1"
                elif font_size > 12:
                    level = "H2"
                elif font_size > 10:
                    level = "H3"

                if level:
                    text = line["text"].strip()
                    # Clean up trailing special characters
                    text = re.sub(r"[:•\*…]+$", "", text).strip()
                    # Remove page numbers like "... 2"
                    text = re.sub(r"\.+\s*\d+$", "", text).strip()
                    # Remove title if present
                    if title:
                        text = text.replace(title, "").strip()

                    if not text:
                        continue

                    page_headings.append({
                        "level": level,
                        "text": text,
                        "page": page_index
                    })

            headings.extend(page_headings)
        
        return title, headings, doc
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return "", [], None

def extract_section_text(doc, heading, next_heading=None):
    """Extract text content for a section based on heading position"""
    if doc is None:
        return ""
        
    start_page = heading["page"]
    text_content = []
    
    # Extract text from the heading's page starting at heading position
    page = doc[start_page]
    blocks = page.get_text("blocks")
    
    # Find the block containing the heading
    start_extracting = False
    for block in blocks:
        # Skip header/footer blocks
        if (block[1] < page.rect.height * 0.1) or (block[3] > page.rect.height * 0.9):
            continue
            
        if not start_extracting and heading["text"] in block[4]:
            start_extracting = True
            # Skip the heading itself since we already have it
            continue
            
        if start_extracting:
            # Stop if we reach the next heading
            if next_heading and next_heading["text"] in block[4]:
                break
            text_content.append(block[4])
    
    return " ".join(text_content)

def preprocess_text(text):
    """Clean and tokenize text for analysis"""
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def rank_sections(documents, persona, job_to_be_done):
    """Rank sections based on relevance to persona and job"""
    # Create a combined query from persona and job
    query = f"{persona['role']} {job_to_be_done['task']}"
    preprocessed_query = preprocess_text(query)
    
    # Collect all section texts for TF-IDF analysis
    section_texts = []
    section_objects = []
    
    for doc_info in documents:
        # Resolve full file path
        filename = os.path.join(BASE_INPUT_PATH, doc_info["filename"])
        title, headings, doc = process_pdf(filename)
        
        # Skip if document failed to process
        if doc is None:
            continue
            
        for i, heading in enumerate(headings):
            next_heading = headings[i+1] if i < len(headings) - 1 else None
            section_text = extract_section_text(doc, heading, next_heading)
            
            # Skip empty sections
            if not section_text.strip():
                continue
                
            # Use the heading text + first 200 chars of section for better relevance
            context_text = f"{heading['text']} {section_text[:200]}"
            preprocessed_text = preprocess_text(context_text)
            
            section_texts.append(preprocessed_text)
            section_objects.append({
                "document": os.path.basename(doc_info["filename"]),
                "section_title": heading['text'],
                "full_text": section_text,
                "page_number": heading['page'] + 1  # Convert to 1-indexed
            })
    
    # Handle case with no valid sections
    if not section_texts:
        return [], []
    
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(section_texts)
        query_vector = vectorizer.transform([preprocessed_query])
    except ValueError:
        # Fallback if no valid terms
        return section_objects, []
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    
    # Rank sections by similarity
    ranked_indices = np.argsort(similarities[0])[::-1]
    ranked_sections = [section_objects[i] for i in ranked_indices]
    
    # Assign importance ranks
    for rank, section in enumerate(ranked_sections, 1):
        section["importance_rank"] = rank
    
    return ranked_sections

def extract_key_sentences(text, min_words=8):
    """Extract meaningful sentences from text"""
    sentences = sent_tokenize(text)
    key_sentences = []
    
    for sentence in sentences:
        # Filter out short, non-meaningful sentences
        if len(sentence.split()) >= min_words:
            # Clean up the sentence
            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
            key_sentences.append(clean_sentence)
            
    return key_sentences

def process_round1b(input_data):
    """Main processing function for Round 1B"""
    # Extract input parameters
    documents = input_data["documents"]
    persona = input_data["persona"]
    job_to_be_done = input_data["job_to_be_done"]
    
    # Process documents and rank sections
    ranked_sections = rank_sections(documents, persona, job_to_be_done)
    
    # Prepare metadata
    metadata = {
        "input_documents": [os.path.basename(doc["filename"]) for doc in documents],
        "persona": persona["role"],
        "job_to_be_done": job_to_be_done["task"],
        "processing_timestamp": datetime.datetime.now().isoformat()
    }
    
    # Prepare extracted sections (top 5)
    extracted_sections = []
    for section in ranked_sections[:5]:
        extracted_sections.append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": section["importance_rank"],
            "page_number": section["page_number"]
        })
    
    # Prepare subsection analysis (key sentences from top sections)
    subsection_analysis = []
    for section in ranked_sections[:5]:
        key_sentences = extract_key_sentences(section["full_text"])
        for sentence in key_sentences:
            subsection_analysis.append({
                "document": section["document"],
                "refined_text": sentence,
                "page_number": section["page_number"]
            })
    
    # Prepare output structure
    output = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis[:5]  # Limit to top 5 sentences
    }
    
    return output

def main_round1b():
    """Entry point for Round 1B processing"""
    print("Starting Round 1B processing...")
    
    # Read input data
    try:
        with open('/app/input/input.json', 'r') as f:
            input_data = json.load(f)
        print("Successfully loaded input.json")
    except Exception as e:
        print(f"Error loading input.json: {str(e)}")
        return
    
    # Process documents
    try:
        output = process_round1b(input_data)
        print(f"Processed {len(input_data['documents'])} documents")
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        return
    
    # Write output
    try:
        with open('/app/output/output.json', 'w') as f:
            json.dump(output, f, indent=2)
        print("Successfully wrote output.json")
    except Exception as e:
        print(f"Error writing output: {str(e)}")

if __name__ == "__main__":
    main_round1b()