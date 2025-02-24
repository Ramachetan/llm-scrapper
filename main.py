import streamlit as st
import asyncio
import os
import sys
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent

load_dotenv()

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def run_agent(user_task):
    sensitive_data = {
        'x_name': os.getenv("LINKEDIN_USERNAME"),
        'x_password': os.getenv("LINKEDIN_PASSWORD")
    }

    agent = Agent(
        task=user_task,
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

    output = {
        "Total Steps": history.number_of_steps(),
        "Total Duration (Seconds)": history.total_duration_seconds(),
        "Total Input Tokens Used": history.total_input_tokens(),
    }

    actions = []
    for step, action in enumerate(history.model_actions(), 1):
        action_serializable = {
            key: (str(value) if not isinstance(value, (dict, list, str, int, float, bool, type(None))) else value)
            for key, value in action.items()
        }
        actions.append({f"Step {step}": action_serializable})
    output["Actions Taken"] = actions

    results = []
    for step, result in enumerate(history.action_results(), 1):
        results.append({f"Step {step}": result.model_dump(exclude_none=True)})
    output["Action Results"] = results

    extracted_content = history.extracted_content()
    output["Extracted Content"] = extracted_content if extracted_content else "No extracted content found."

    errors = history.errors()
    output["Errors"] = errors if any(errors) else "No errors encountered."

    final_result = history.final_result()
    output["Final Result"] = final_result if final_result else "No final result available."
    output["Success Status"] = "✔️ Task Completed Successfully" if history.is_successful() else "❌ Task Failed or Incomplete."

    return output

def main():
    st.title("LLM Scraper Agent")
    st.write("Enter a task for the agent to execute.")

    user_task = st.text_area(
        "Enter Task",
        "Go to Google, search for 'browser-use', click the first result, and return findings in JSON."
    )

    if st.button("Run Agent"):
        if not user_task.strip():
            st.error("Please enter a valid task.")
        else:
            status_placeholder = st.empty()
            status_placeholder.info("Running agent...")
            result = asyncio.run(run_agent(user_task))
            status_placeholder.empty()
            st.success("Agent execution completed!")
            
            st.subheader("Data Extracted")
            extracted_content = result.get("Extracted Content", "")

            with st.expander("Summary"):
                st.markdown(f"**Total Steps:** {result.get('Total Steps', 'N/A')}")
                st.markdown(f"**Total Duration (Seconds):** {result.get('Total Duration (Seconds)', 'N/A')}")
                st.markdown(f"**Total Input Tokens Used:** {result.get('Total Input Tokens Used', 'N/A')}")
                st.markdown(f"**Success Status:** {result.get('Success Status', 'N/A')}")

            with st.expander("Actions"):
                st.subheader("Actions Taken")
                for action in result.get("Actions Taken", []):
                    st.json(action)

                st.subheader("Action Results")
                for action_result in result.get("Action Results", []):
                    st.json(action_result)
            
            with st.expander("Errors"):
                st.subheader("Errors")
                st.write(result.get("Errors", "No errors encountered."))
                
            if isinstance(extracted_content, list):
                extracted_content = "\n".join(extracted_content)

            pattern = r"```json\s*(.*?)```"
            json_blocks = re.findall(pattern, extracted_content, re.DOTALL)

            if json_blocks:
                for idx, json_block in enumerate(json_blocks, 1):
                    st.markdown(f"**Extracted JSON Block {idx}:**")
                    try:
                        json_data = json.loads(json_block)
                        st.json(json_data)
                    except Exception as e:
                        st.warning("Failed to parse JSON block. Displaying raw content:")
                        st.code(json_block, language="json")
              
                cleaned_content = re.sub(pattern, '', extracted_content, flags=re.DOTALL).strip()
                if cleaned_content:
                    with st.expander("Other Extracted Content"):
                        st.markdown("**Other Extracted Content:**")
                        st.write(cleaned_content)
            else:
                st.write(extracted_content)

            st.subheader("Final Response")
            st.write(result.get("Final Result", "No final result available."))

if __name__ == "__main__":
    main()
