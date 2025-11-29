import os
import json
import logging
import requests
import nest_asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from playwright.async_api import async_playwright
from openai import OpenAI

# Fix for asyncio loops in some environments
nest_asyncio.apply()

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# 1. Your Secret (Must match what you put in Google Form)
MY_SECRET = "tds_secret_key" 

# 2. Setup AI Pipe (The IITM Proxy)
# We get the key from Render Environment Variables for security
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), 
    base_url="https://aipipe.org/openai/v1"
)

class QuizTask(BaseModel):
    email: str
    secret: str
    url: str

@app.get("/")
def home():
    return {"message": "tripti's AI Agent is Ready."}

@app.post("/solve")
async def solve_quiz(task: QuizTask, background_tasks: BackgroundTasks):
    # 1. Verification
    if task.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # 2. Start the Agent in Background (so we return 200 OK immediately)
    background_tasks.add_task(run_agent_logic, task.url, task.email)
    
    return {"message": "Agent started", "url": task.url}

async def run_agent_logic(start_url: str, email: str):
    logger.info(f"Starting agent on: {start_url}")
    current_url = start_url
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Use a fake User-Agent so we look like a normal Chrome browser
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()

        # Loop through questions (max 5 to prevent infinite loops)
        for _ in range(5):
            if not current_url:
                break
                
            logger.info(f"Navigating to: {current_url}")
            try:
                await page.goto(current_url, timeout=60000)
                await page.wait_for_load_state("networkidle")
                
                # 1. Scrape the visible text
                body_text = await page.evaluate("document.body.innerText")
                
                # 2. Ask GPT-4o-mini to solve it
                prompt = f"""
                You are an intelligent agent solving a data quiz.
                
                PAGE CONTENT:
                ----------------
                {body_text}
                ----------------
                
                YOUR TASKS:
                1. Identify the SUBMISSION URL mentioned in the text (e.g., https://example.com/submit).
                2. Solve the question asked in the text.
                3. Return a JSON object with:
                   - "answer": (the answer value)
                   - "submit_url": (the extracted URL to post to)
                
                Strictly return ONLY valid JSON. No markdown.
                """
                
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Parse AI response
                ai_response = completion.choices[0].message.content.strip()
                if "```json" in ai_response:
                    ai_response = ai_response.split("```json")[1].split("```")[0].strip()
                elif "```" in ai_response:
                    ai_response = ai_response.split("```")[1].split("```")[0].strip()
                
                data = json.loads(ai_response)
                answer_val = data.get("answer")
                submit_url = data.get("submit_url")
                
                logger.info(f"AI decided: Answer={answer_val}, SubmitTo={submit_url}")

                if not submit_url:
                    logger.error("No submit URL found by AI.")
                    break

                # 3. Submit the answer
                payload = {
                    "email": email,
                    "secret": MY_SECRET,
                    "url": current_url,
                    "answer": answer_val
                }
                
                # Post to the quiz server
                resp = requests.post(submit_url, json=payload, timeout=10)
                logger.info(f"Submission Response: {resp.status_code} - {resp.text}")
                
                resp_json = resp.json()
                
                # 4. Check if we need to go to the next question
                if resp_json.get("correct", False) is True:
                    next_url = resp_json.get("url")
                    if next_url:
                        current_url = next_url # Loop continues with new URL
                    else:
                        logger.info("Quiz completed successfully!")
                        break
                else:
                    logger.warning("Answer was incorrect. Stopping.")
                    break

            except Exception as e:
                logger.error(f"Error in processing: {e}")
                break
        
        await browser.close()
