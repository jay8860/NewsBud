# Intelligent Newspaper Editorial Summarizer Bot

A Telegram bot that uses Gemini's multimodal capabilities to automatically detect, extract, and summarize editorial pages from newspaper PDFs.

## Features

1.  **Dynamic Page Detection**: Scans the first 12 pages of a PDF to find "Editorial", "Opinion", or "Ideas" sections using Gemini 1.5 Flash.
2.  **High-Res Analysis**: Extracts identified pages at 300 DPI for optimal legibility.
3.  **Smart Summarization**: Generates a "Decision-Maker’s Brief" for key articles, including Core Argument, Key Evidence, and Policy Implications.
4.  **Fallback Mechanism**: Allows manual page entry via `/pages` command if detection fails.

## Setup & Deployment

### Local Development

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `.env` file (copy from `.env.example`) and add your keys:
    ```ini
    TELEGRAM_BOT_TOKEN=your_token
    GEMINI_API_KEY=your_key
    ```
4.  Run the bot:
    ```bash
    python bot.py
    ```

### Deploy to Railway.app

1.  Fork/Clone this repository to your GitHub.
2.  Login to [Railway.app](https://railway.app/).
3.  Click **New Project** -> **Deploy from GitHub repo**.
4.  Select this repository.
5.  Railway will automatically detect the `Dockerfile`.
6.  Go to the **Variables** tab in your Railway service dashboard.
7.  Add the following environment variables:
    *   `TELEGRAM_BOT_TOKEN`
    *   `GEMINI_API_KEY`
8.  The bot will deploy and start automatically.

## Requirements

*   A Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
*   A Google Gemini API Key (from [Google AI Studio](https://aistudio.google.com/))
