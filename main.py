from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    agent = Agent(
        task="Go to google, search for 'browser-use', click on the first link and return the results",
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME"),
            temperature=0.2,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        ),
    )
    result = await agent.run()
    print(result)

asyncio.run(main())