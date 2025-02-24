from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
import asyncio
import os
import json
from dotenv import load_dotenv

load_dotenv()

async def main():
    sensitive_data = {
        'x_name': os.getenv("LINKEDIN_USERNAME"), 
        'x_password': os.getenv("LINKEDIN_PASSWORD")
    }

    print("Sensitive Data (Hidden for security):", {k: "******" for k in sensitive_data.keys()})
    
    agent = Agent(
        task="Go to Google, search for 'langchain', click the first result, and return findings in JSON.",
        llm=ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        ),
        use_vision=True,
        save_conversation_path="logs/conversation",
        sensitive_data=sensitive_data,
    )

    history = await agent.run()

    # Printing everything from history
    print("\n===== AGENT EXECUTION HISTORY =====\n")

    print("🔹 Total Steps:", history.number_of_steps())
    print("🔹 Total Duration (Seconds):", history.total_duration_seconds())
    print("🔹 Total Input Tokens Used:", history.total_input_tokens())

    print("\n===== ACTIONS TAKEN BY AGENT =====")
    for step, action in enumerate(history.model_actions(), 1):
        # Convert non-serializable objects to string representations
        action_serializable = {
            key: (str(value) if isinstance(value, object) and not isinstance(value, (dict, list, str, int, float, bool, type(None))) else value)
            for key, value in action.items()
        }
        print(f"\nStep {step}:")
        print(json.dumps(action_serializable, indent=2))

    print("\n===== RESULTS OF ACTIONS =====")
    for step, result in enumerate(history.action_results(), 1):
        print(f"\nStep {step}:")
        print(json.dumps(result.model_dump(exclude_none=True), indent=2))

    print("\n===== EXTRACTED CONTENT =====")
    extracted_content = history.extracted_content()
    if extracted_content:
        for step, content in enumerate(extracted_content, 1):
            print(f"\nStep {step} Content:\n{content}")
    else:
        print("No extracted content found.")

    print("\n===== ERRORS (IF ANY) =====")
    errors = history.errors()
    if any(errors):
        for step, error in enumerate(errors, 1):
            if error:
                print(f"\nStep {step} Error:\n{error}")
    else:
        print("No errors encountered.")

    print("\n===== FINAL RESULT =====")
    final_result = history.final_result()
    print(final_result if final_result else "No final result available.")

    print("\n===== SUCCESS STATUS =====")
    print("✔️ Task Completed Successfully?" if history.is_successful() else "❌ Task Failed or Incomplete.")

asyncio.run(main())
