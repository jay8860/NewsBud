import os
import logging
import tempfile
import asyncio
import time
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
MODEL_NAME = "gemini-2.0-flash" 

async def process_pdf(pdf_path: str) -> str:
    """Uploads PDF to Gemini and gets the editorial analysis."""
    logger.info(f"Uploading file to Gemini: {pdf_path}")
    
    # Upload file
    # We must run this in a thread because it's synchronous IO
    loop = asyncio.get_running_loop()
    
    # Define a helper for the blocking upload call
    def upload_and_wait():
        sample_file = genai.upload_file(pdf_path, mime_type="application/pdf")
        # Wait for processing to complete. 
        # For small PDFs it's instant, but safe to check.
        while sample_file.state.name == "PROCESSING":
             time.sleep(1)
             sample_file = genai.get_file(sample_file.name)
        return sample_file

    gemini_file = await loop.run_in_executor(None, upload_and_wait)
    
    if gemini_file.state.name == "FAILED":
        raise ValueError("Gemini failed to process the PDF file.")
        
    logger.info(f"File uploaded: {gemini_file.name}")
    
    # Generate content
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = """
    You are an expert policy analyst. Using the attached newspaper document:
    
    1.  **Locate** the "Editorial", "Opinion", or "Ideas" section (usually in the middle pages).
    2.  **Identify** the 3-4 most important articles within that section.
    3.  **Generate a "Decision-Maker’s Brief"** for each, with:
        *   **Title** of the Article.
        *   **Core Argument:** (2-3 sentences summarizing the main point).
        *   **Key Data/Evidence:** (Bullet points of specific stats, names, or evidence cited).
        *   **Policy Implications:** (Relevance for a senior government official).
    
    Format the output in clean Markdown with bold headers and bullet points.
    If you cannot find the editorials, say "Could not locate editorial section."
    """
    
    response = await model.generate_content_async([gemini_file, prompt])
    
    # Cleanup (Optional but good practice to delete file from cloud storage)
    # genai.delete_file(gemini_file.name) # Deferring for now to keep it simple/fast
    
    return response.text

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome! Send me a newspaper PDF, and I'll analyze the editorials instantly."
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.document.mime_type == 'application/pdf':
        await update.message.reply_text("Please send a PDF file.")
        return

    status_msg = await update.message.reply_text("Processing PDF... (Uploading to Brain)")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download file
            logger.info(f"Downloading file: {update.message.document.file_id}")
            file = await context.bot.get_file(update.message.document.file_id)
            pdf_path = os.path.join(temp_dir, "newspaper.pdf")
            await file.download_to_drive(pdf_path)
            
            # Process with Gemini
            await status_msg.edit_text("Reading newspaper (Gemini 2.0 Flash is analyzing)...")
            summary = await process_pdf(pdf_path)
            
            await status_msg.edit_text("Here is your Decision-Maker’s Brief:")
            await update.message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)
            logger.info("Process complete.")

    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        await status_msg.edit_text(f"An error occurred: {str(e)}")

def main():
    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        print("Error: TELEGRAM_BOT_TOKEN and GEMINI_API_KEY must be set in .env file.")
        return

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))

    print(f"Bot is running with {MODEL_NAME}...")
    application.run_polling()

if __name__ == '__main__':
    main()
