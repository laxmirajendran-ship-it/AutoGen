import asyncio
import os
import json
import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient


# ----------------------------
# FILE WRITER TOOLS
# ----------------------------
def write_file(content: str, type: Literal["test_case", "step_definition"] = "test_case") -> dict:
    """Writes test cases or step definitions to disk."""
    work_dir = Path.cwd() / "outputs"
    work_dir.mkdir(exist_ok=True)

    ts = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
    filename = (
        f"Test_Cases_{ts}.csv" if type == "test_case"
        else f"steps_{ts}.py"
    )

    path = work_dir / filename
    path.write_text(content, encoding="utf-8")
    return {"path": str(path)}

write_file_tool = FunctionTool(
    write_file,
    name="write_file",
    description="Save CSV test cases or Python step definitions."
)


def write_java_file(filename: str, content: str) -> dict:
    work_dir = Path.cwd() / "outputs" / "java"
    work_dir.mkdir(parents=True, exist_ok=True)

    path = work_dir / filename
    path.write_text(content, encoding="utf-8")
    return {"path": str(path)}

write_java_tool = FunctionTool(
    write_java_file,
    name="write_java_file",
    description="Write Java Step Definition file. Args: filename, content",
)


# ----------------------------
# MAIN AUTOGEN WORKFLOW
# ----------------------------
async def run_autogen_workflow(requirements: str):
    import streamlit as st

    load_dotenv()

    def get_requirements_input(prompt: str) -> str:
        return requirements

    # Load config.json configuration
    config_path = Path(__file__).parent / "config.json"
    config = json.loads(Path(config_path).read_text())

    # Choose LLM client
    if config.get("use_ollama"):
        model_client = OllamaChatCompletionClient(
            model=config.get("ollama_model_name"),
            temperature=config.get("temperature", 0.1) ,
            model_capabilities={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "structured_output": False
            }
        )
    else:
        model_client = OpenAIChatCompletionClient(
            model=config.get("openai_model_name"),
            temperature=config.get("temperature", 0.1),
            model_capabilities={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "structured_output": False
            },
            api_key=config.get("openai_api_key") if config.get("openai_api_key") != "" else os.environ.get("OPENAI_API_KEY")
        )

    # -------------------------
    # USER AGENT
    # -------------------------
    user_proxy = UserProxyAgent(
        "user_proxy",
        input_func=get_requirements_input,
        
        # This agent does not need a system message as it only provides the initial prompt.
    )

    # -------------------------
    # TEST MANAGER (creates Gherkin)
    # -------------------------
    TestManager = AssistantAgent(
        name="TestManager",
        model_client=model_client,
        system_message="""
You are a Test Manager. 
1. Convert requirements into a User Story.
2. Produce Acceptance Criteria.
3. Output everything in proper Gherkin syntax (Feature/Scenario/Given/When/Then).
Do NOT request user input again.
"""
    )

    # -------------------------
    # TEST CASE WRITER
    # -------------------------
    test_case_writer = AssistantAgent(
        name="test_case_writer",
        model_client=model_client,
        tools=[write_file_tool],
        system_message="""
You write detailed test cases in CSV format.

Columns:
1. Test Case ID
2. Test Case Name
3. Preconditions
4. Test Steps (Numbered)
5. Expected Result

Once generated, call the write_file tool:
write_file(content=<csv>, type="test_case")
"""
    )

    def write_java_file(filename: str, content: str) -> dict:
        print("Writing Java Step Definition:", filename)
        work_dir = Path.cwd() / "outputs" / "java"
        work_dir.mkdir(parents=True, exist_ok=True)
        p = work_dir / filename
        p.write_text(content, encoding="utf-8")
        return {"path": str(p)}

    write_java_tool = FunctionTool(
        write_java_file,
        name="write_java_file",
        description="Write Java StepDefinition file. Args: filename, content",
    )


    # Step 4: An agent called "Test case reviewer" shares the feedback
    test_case_reviewer = AssistantAgent(
        "test_case_reviewer",
        model_client=model_client,
        system_message=(
            "You are a meticulous test case reviewer. Your task is to review the test case CSV file created by the 'test_case_writer'. "
            "Refer to best practices mentioned in the documents in the './docs' folder.\n"
            "Provide constructive feedback on the file. The 'test_case_writer' agent will then provide a revised version in CSV format .\n"
            "If the test cases are satisfactory and meet all criteria, reply with 'APPROVED'. Otherwise, provide specific suggestions for changes."
        )
    )

    # Step 5: An agent called BDD coder writes stepdefinition in python
    bdd_coder = AssistantAgent(
        "bdd_coder",
        model_client=model_client,
        system_message=(
            "You are an expert BDD Coder. You write step definitions in Python using the 'behave' library syntax.\n"
            # "IMPORTANT: Do not do anything until you see the message 'APPROVED' from the 'test_case_reviewer' in the conversation history.\n"
            # "Once you see 'APPROVED',
            "take the Gherkin Feature content provided by the TestManager and generate the corresponding step definition code.\n"
            "Ensure all 'Given', 'When', 'And', 'But', and 'Then' steps are implemented with a basic 'pass' statement.\n"
            "Make sure to import necessary modules from 'behave'.\n"
            "Save the step definitions in a Python file named 'steps.py' using the 'write_file' tool.\n"
            "Reply with 'TERMINATE' only after the Python file has been successfully written."
        ),
        tools=[write_file_tool],
    )

    step_definition_agent = AssistantAgent(
        "step_definition_agent",
        model_client=model_client,
        system_message=(
            "Your task is to generate Java Selenium+Cucumber Step Definitions taking input as UserStory and Acceptance Criteria generated by the TestManager agent."
            "Rules:\n"
            "- Output MUST be a full Java class named StepDefinition.java\n"
            "- Use @Given, @When, @Then, @And\n"
            "- Convert steps to Java methods with camelCase names\n"
            "- Add Selenium logic using driver.findElement(By.id/xpath).click(), sendKeys, getText\n"
            "- No markdown. Only pure Java code.\n"
            "- After generating code, call write_java_file(filename='StepDefinition.java', content=<code>)" 
            "and name the java file in the format 'StepDefinition_MMMDDYYYY_HHMMSS.java' in outputs/java folder."
        ),
        tools=[write_java_tool],
    )
    # The looping between writer and reviewer is handled by the group chat dynamics.
    # The BDD coder acts after the initial Gherkin is produced.
    agent_team = RoundRobinGroupChat(
       [user_proxy, TestManager, test_case_writer, step_definition_agent],
       max_turns=4 # Increased max_turns to allow for review loops
    )

    try:
        # The initial task for the team.
        task = "Create a user story, acceptance criteria, test cases based on the user's requirement."
        
        async for message in agent_team.run_stream(task=task):
            if isinstance(message, str):
                yield {"source": "System", "content": message}
                continue

Otherwise provide clear feedback.
"""
    )

    # -------------------------
    # STEP DEFINITION AGENT (Java)
    # -------------------------
    step_definition_agent = AssistantAgent(
        name="step_definition_agent",
        model_client=model_client,
        tools=[write_java_tool],
        system_message="""
Generate Java Selenium+Cucumber step definitions.

Rules:
- Output only valid Java.
- Class must be named StepDefinition.
- Use @Given/@When/@Then annotations.
- Convert Gherkin steps into Java methods.
- Use Selenium driver.findElement(...) examples.
- Afterwards call:

write_java_file(filename="StepDefinition.java", content=<java_code>)
"""
    )

    # -------------------------
    # TEAM ORCHESTRATION
    # -------------------------
    # Get selected agents from session state
    selected_agents = [user_proxy]  # Start with user_proxy

    # Always include TestManager if either user_story_writer or test_case_writer is selected
    if st.session_state.agents.get('user_story_writer', False) or st.session_state.agents.get('test_case_writer', False):
        selected_agents.append(TestManager)
    
    # Add other agents based on selection
    if st.session_state.agents.get('test_case_writer', False):
        selected_agents.append(test_case_writer)
    if st.session_state.agents.get('step_definition_writer', False):
        selected_agents.append(step_definition_agent)
        
    
    # Set max_turns based on number of agents
    max_turns = len(selected_agents)
    team = RoundRobinGroupChat(
        selected_agents,
        max_turns=max_turns
    )

    # -------------------------
    # RUN & STREAM OUTPUT
    # -------------------------
    try:
        async for message in team.run_stream(task=requirements):
            if hasattr(message, "source") and isinstance(message.source, str) and message.source == "user":
                continue

            # If the message came from a tool, show the file path
            if getattr(message, "role", "") == "tool":
                try:
                    data = json.loads(message.content)
                    if "path" in data:
                        yield {
                            "source": message.name,
                            "content": f"File created: {data['path']}",
                            "path": data["path"]
                        }
                except:
                    pass
                continue

            # Stream normal messages
            if hasattr(message, "content") and isinstance(message.content, str):
                yield {"source": message.source, "content": message.content}

    except Exception as e:
        yield {"source": "System", "content": f"ERROR: {e}"}

    finally:
        await model_client.close()
