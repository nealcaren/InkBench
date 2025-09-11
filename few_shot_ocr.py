## To run this code you need to install the following dependencies:
# pip install requests

import requests
import base64
from pathlib import Path
import os 
import mimetypes


def transcribe_historical_document(image_path: str, api_key: str = None, samples_folder: str = "samples", model: str = "google/gemma-3-27b-it") -> str:
    """
    Transcribe a historical document image using OpenRouter API with few-shot learning examples.
    
    Args:
        image_path (str): Path to the image file to transcribe
        api_key (str, optional): OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var
        samples_folder (str): Path to folder containing sample images and their transcriptions
        model (str): Model to use for transcription (default: "google/gemma-3-27b-it")
        
    Returns:
        str: Transcribed text from the document
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If API key is not provided
        requests.RequestException: If API call fails
    """
    
    # System prompt for transcription guidelines
    system_prompt = '''
# General Transcription Guide
Following these directions when transcribing the text in the image.

## Core Principles

### Transcribe What You See
- Preserve original spelling, grammar, and punctuation exactly as written
- Do not paraphrase or correct errors
- Type what you see, not what you think it should say
- Transcribe text in the order it appears on the page (as you would read it aloud)

### When Not to Transcribe
- **Blank pages**: Check "Nothing to Transcribe" and save
- **Images only**: If a page contains only images and no text, click "Nothing to Transcribe"
- **Mass-produced content**: Skip pre-printed calendars, almanacs, and dates in diaries that don't aid historical discovery
- **Shorthand**: Insert [[shorthand]] instead of attempting to transcribe shorthand symbols

## Text Handling

### Basic Text
- Transcribe all legible text including letterheads, page numbers, and catalog numbers
- Read and transcribe columns in logical reading order
- Don't attempt to preserve formatting (bold, italic, underline, indentation)

### Abbreviations and Symbols
- Do not expand abbreviations - transcribe exactly as written
- Transcribe symbols as they appear (&, $, £, §, etc.)
- Use lowercase "s" for historical "long s" that looks like "f"

### Line and Page Breaks
- Preserve line breaks as they appear
- When words break across lines on the same page, write the complete word on the first line
- When words break across pages, transcribe the word on the first page
- For multiple pages in one image, transcribe all pages in order with empty lines between them

## Special Situations

### Illegible or Unclear Text
- Use [?] for completely illegible words
- For partially readable words, transcribe visible letters: s[????] for "s" followed by unreadable letters
- If you can't read much of a page, save and move to another document

### Text Modifications
- **Deletions**: Transcribe crossed-out text within square brackets [deleted text]
- **Insertions**: Include inserted text in natural reading order without special notation
- **Marginalia**: Transcribe margin notes within [* *], placed after relevant text or at the end if unrelated

### Punctuation and Special Cases
- Make best judgment for historical punctuation (en dash, em dash, etc.)
- Reduce excessive ellipses to three dots (...)
- Ignore bleed-through text (backward/mirror text from other side of thin paper)

## Document-Specific Guidelines

### Tables
- Preserve relationships between columns and rows
- Focus on capturing data accurately, not replicating exact layout
- Use spaces and line returns to maintain readability

### Newspapers
- Transcribe all articles, not just seemingly relevant ones
- Follow natural column reading order
- Don't attempt to preserve layout

### Envelopes
- Transcribe all legible text including postmarks and stamps
- Mark postmarks and stamps as marginalia [* *]

### Cross-Writing
- Transcribe layered text in natural reading order
- Add "cross-writing" tag when applicable

## Languages and Characters

### Non-English Text
- Transcribe completely in original language
- Use correct characters (Ç not C)
- Do not translate - transcribe only
- Use alt codes for special characters

## Quality Guidelines

### What to Avoid
- Don't describe images or non-text features in transcription
- Don't add formatting characters or layout symbols
- Don't guess at heavily damaged or illegible text

'''

    def encode_image_to_base64(image_path: str) -> str:
        """Encode a local image file as a base64 data URL (with mime prefix)."""
        mime, _ = mimetypes.guess_type(image_path)
        if mime is None:
            mime = "image/jpeg"  # fallback
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    # Validate inputs
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("API key must be provided either as parameter or OPENROUTER_API_KEY environment variable")

    # Load sample images and transcriptions for few-shot learning
    def load_samples(samples_folder: str):
        """Load sample images and their transcriptions from the samples folder."""
        samples_path = Path(samples_folder)
        if not samples_path.exists():
            print(f"Warning: Samples folder '{samples_folder}' not found. Proceeding without examples.")
            return []
        
        samples = []
        # Look for image files in the samples folder
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff"]
        for pattern in image_extensions:
            for img_file in samples_path.glob(pattern):
                txt_file = img_file.with_suffix(".txt")
                if txt_file.exists():
                    try:
                        img_encoded = encode_image_to_base64(str(img_file))
                        with open(txt_file, "r", encoding="utf-8") as f:
                            transcription = f.read()
                        samples.append((img_encoded, transcription))
                    except Exception as e:
                        print(f"Warning: Failed to load sample {img_file}: {e}")
        
        return samples

    # Encode the target image
    try:
        target_image_encoded = encode_image_to_base64(image_path)
    except Exception as e:
        raise ValueError(f"Failed to encode image: {e}")

    # Load sample examples
    sample_examples = load_samples(samples_folder)

    # Prepare API request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Build messages with system prompt and few-shot examples
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    
    # Add sample examples as conversation history
    for sample_img, sample_text in sample_examples:
        messages.extend([
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Provide a complete transcript of this item"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": sample_img
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample_text
                    }
                ]
            }
        ])
    
    # Add the actual image to transcribe
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Provide a complete transcript of this item"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": target_image_encoded
                }
            }
        ]
    })
    
    payload = {
        "model": model,
        "messages": messages
    }

    # Make API call
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        result = response.json()
        # print(result)
        return result["choices"][0]["message"]["content"]
        
    except requests.RequestException as e:
        raise requests.RequestException(f"API call failed: {e}")
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected API response format: {e}")

'''
transcription = transcribe_historical_document("/Users/nealcaren/Downloads/InkBench/early-copyright_2025-06-02/2019713420-329.jpg",
                                               model="google/gemini-2.5-flash")
print(transcription)
'''