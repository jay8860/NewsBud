import os
import io
import logging
import tempfile
import json
import asyncio
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    
# Constants
DETECTION_MODEL_NAME = "gemini-1.5-flash"
ANALYSIS_MODEL_NAME = "gemini-1.5-flash" # Can trigger to pro if needed, but flash is good for text analysis too. Let's use flash for speed/cost unless quality is poor.
MAX_THUMBNAIL_PAGES = 12

def render_thumbnails(pdf_path: str, max_pages: int = 12) -> List[Image.Image]:
    """Generates low-res thumbnails for the first few pages of a PDF."""
    doc = fitz.open(pdf_path)
    images = []
    num_pages = min(doc.page_count, max_pages)
    
    for i in range(num_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=72) # Low res for detection
        img_data = pix.tobytes("png")
        images.append(Image.open(io.BytesIO(img_data)))
        
    doc.close()
    return images

def render_high_res(pdf_path: str, page_numbers: List[int]) -> List[Image.Image]:
    """Generates high-res images for specific pages."""
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in page_numbers:
        # page_num is 1-based from user/gemini, convert to 0-based
        if 1 <= page_num <= doc.page_count:
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=300) # High res for OCR/Reading
            img_data = pix.tobytes("png")
            images.append(Image.open(io.BytesIO(img_data)))
            
    doc.close()
    return images

async def detect_editorial_pages(images: List[Image.Image]) -> List[int]:
    """Uses Gemini to identify editorial pages from thumbnails."""
    model = genai.GenerativeModel(DETECTION_MODEL_NAME)
    
    prompt = """
    Analyze these newspaper pages. Identify the page numbers that contain the "Editorial", "Opinion", or "Ideas" sections.
    Return ONLY a JSON array of integers representing the page numbers (1-based index corresponding to the order of images provided).
    Example: [6, 7]
    If none found, return [].
    """
    
    try:
        response = await model.generate_content_async([prompt, *images])
        text = response.text.strip()
        # Clean up code blocks if present
        if text.startswith("```json"):
            text = text[7:-3]
        elif text.startswith("```"):
            text = text[3:-3]
            
        return json.loads(text)
    except Exception as e:
        logger.error(f"Error in page detection: {e}")
        return []

async def analyze_pages(images: List[Image.Image]) -> str:
    """Uses Gemini to summarize the editorial pages."""
    model = genai.GenerativeModel(ANALYSIS_MODEL_NAME)
    
    prompt = """
    You are an expert policy analyst. Analyze these high-resolution images of newspaper editorial pages.
    Identify the 3-4 most important articles.
    
    For each article, generate a "Decision-Maker’s Brief" with the following format:
    
    ### [Title of the Article]
    *   **Core Argument:** (2-3 sentences summarizing the main point)
    *   **Key Data/Evidence:** (Bullet points of specific stats, names, or evidence cited)
    *   **Policy Implications:** (Relevance for a senior government official)
    
    Format the output in clean Markdown. Use bold headers and bullet points.
    """
    
    try:
        response = await model.generate_content_async([prompt, *images])
        return response.text
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return "Error analyzing the pages. Please try again."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome! Send me a newspaper PDF, and I'll extract and summarize the editorials for you."
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.document.mime_type == 'application/pdf':
        await update.message.reply_text("Please send a PDF file.")
        return

    status_msg = await update.message.reply_text("Processing PDF... generating thumbnails.")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download file
        file = await context.bot.get_file(update.message.document.file_id)
        pdf_path = os.path.join(temp_dir, "newspaper.pdf")
        await file.download_to_drive(pdf_path)
        
        # Step 1: Dynamic Detection
        thumbnails = render_thumbnails(pdf_path)
        await status_msg.edit_text("Scanning for editorial pages...")
        
        detected_pages = await detect_editorial_pages(thumbnails)
        
        if not detected_pages:
            await status_msg.edit_text(
                "Could not automatically locate editorials. Please reply with the page numbers manually using the /pages command.\n"
                "Example: `/pages 6 7`",
                parse_mode=ParseMode.MARKDOWN
            )
            # Store pdf_path in user_data is difficult across distinct updates due to temp dir cleanup.
            # Ideally we'd persist the file. For simplicity in this script, we'll ask user to re-upload or implement a file persistence if needed.
            # But the requirement says "reply with page numbers". 
            # To support "reply", we need to keep the file. 
            # Let's move the file to a persistent temp location renamed by file_id to handle the fallback.
            
            persist_dir = os.path.join(tempfile.gettempdir(), "editorial_bot_files")
            os.makedirs(persist_dir, exist_ok=True)
            persist_path = os.path.join(persist_dir, f"{update.message.document.file_id}.pdf")
            import shutil
            shutil.copy(pdf_path, persist_path)
            
            context.user_data['last_pdf_path'] = persist_path
            return

        await status_msg.edit_text(f"Found editorials on pages {detected_pages}. Analyzing...")
        
        # Step 2: High-Res Extraction
        high_res_images = render_high_res(pdf_path, detected_pages)
        
        # Step 3: Analytical Summarization
        summary = await analyze_pages(high_res_images)
        
        await status_msg.edit_text("Here is your Decision-Maker’s Brief:")
        await update.message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)

async def pages_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Please provide page numbers. Example: /pages 6 7")
        return
        
    last_pdf_path = context.user_data.get('last_pdf_path')
    if not last_pdf_path or not os.path.exists(last_pdf_path):
        await update.message.reply_text("No recent PDF found. Please upload the newspaper again.")
        return
        
    try:
        pages = [int(arg) for arg in context.args]
        status_msg = await update.message.reply_text(f"Analyzing pages {pages}...")
        
        high_res_images = render_high_res(last_pdf_path, pages)
        summary = await analyze_pages(high_res_images)
        
        await status_msg.edit_text("Here is your Decision-Maker’s Brief:")
        await update.message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)
        
        # Cleanup
        try:
            os.remove(last_pdf_path)
            del context.user_data['last_pdf_path']
        except:
            pass
            
    except ValueError:
        await update.message.reply_text("Invalid page numbers. Please use format: /pages 6 7")

def main():
    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        print("Error: TELEGRAM_BOT_TOKEN and GEMINI_API_KEY must be set in .env file.")
        return

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    application.add_handler(CommandHandler("pages", pages_command))

    print("Bot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()
