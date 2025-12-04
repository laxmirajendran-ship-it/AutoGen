from dotenv import load_dotenv
import asyncio
from pathlib import Path
import os
import datetime
import json
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from typing import Literal
# Function to write output to a file
def write_file(content: str, type: Literal["test_case", "step_definition"] = "test_case") -> dict:
    work_dir = Path.cwd() / "outputs"
    work_dir.mkdir(exist_ok=True)
    if type == "test_case":
        filename = f"Test_Cases_{datetime.datetime.now().strftime('%m%d%Y_%H%M%S')}.csv"
    else:
        filename = f"steps_{datetime.datetime.now().strftime('%m%d%Y_%H%M%S')}.py"
    p = work_dir / filename
    p.write_text(content, encoding="utf-8")
    return {"path": str(p)}

write_file_tool = FunctionTool(
    write_file,
    name="write_file",
    description="Write given content to a file in the current working directory. Args: content, type (test_case or step_definition)",
)

class SimpleLocalCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
    async def start(self):
        return None

    async def stop(self):
        return None

async def run_autogen_workflow(requirements: str):
    load_dotenv()
    """
    This function orchestrates the multi-agent workflow.
    """
    # This inner function is used by the UserProxyAgent to get the initial user requirement.
    def get_requirements_input(prompt: str) -> str:
        return requirements
    
    # Load configuration from config.json
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

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


    # This agent is the entry point, representing the user who provides the initial requirement.
    user_proxy = UserProxyAgent(
        "user_proxy",
        input_func=get_requirements_input,
        # This agent does not need a system message as it only provides the initial prompt.
    )

    # Step 2: An agent called 'Test Manager" writes a detailed userstory and acceptance criteria
    TestManager = AssistantAgent(
        "TestManager",
        model_client=model_client,
        system_message="""You are a TestManager. Your first task is to take the requirements provided and create a User Story and Acceptance Criteria.
Create a User Story and Acceptance Criteria from the given requirements.
The User Story and Acceptance Criteria must be written in a well-structured Gherkin syntax.
Ensure you use the correct Gherkin keywords: Feature, Scenario, Given, When, Then, And, But.
After providing the Gherkin content, ask the 'test_case_writer' to proceed. Do not ask for user input again.
Do not use termination phrases like 'TERMINATE'.""",
    )

    # Step 3: An agent called 'Test case writer" writes detailed test cases
    test_case_writer = AssistantAgent(
        "test_case_writer",
        model_client=model_client,
        system_message=(
            "You are a skilled test case writer. Your task is to create detailed and effective test cases in CSV format "
            "based on the User Story and Acceptance Criteria provided by the TestManager. "
            "Refer to examples and guidelines in the './docs' folder for best practices.\n"
            "First, generate a unique filename for the test cases using the current timestamp (e.g., 'Test_Cases_YYYYMMDD_HHMMSS.csv').\n"
            "Then, produce the test cases in CSV format with the following columns in the exact order:\n"
            "1. Test Case ID (e.g., TC_001)\n"
            "2. Test Case Name\n"
            "3. Preconditions\n"
            "4. Test Steps (Numbered steps)\n"
            "5. Expected Result\n\n"
            "Ensure each column is properly populated. "
            "Finally, use the 'write_file' tool to save the test cases to the generated CSV file. "
            "After saving the file, ask the 'test_case_reviewer' to review the file you have created."
        ),
        tools=[write_file_tool],
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

            # The message is a dictionary.
            source_name = message.name if hasattr(message, 'name') else "System"
            content = message.content if hasattr(message, 'content') else None
            role = message.role if hasattr(message, 'role') else "user"


            if not source_name or content is None:
                continue

            if not isinstance(content, str):
                continue

            content = content.strip()
            if not content:
                continue

            # Check for tool call results (file paths)
            if role == "tool":
                try:
                    data = json.loads(content)
                    if "path" in data:
                        yield {"source": "System", "content": f"File created: {data['path']}", "path": data['path']}
                except (json.JSONDecodeError, TypeError):
                    # Not a valid JSON, or content is not a string, stream as is
                    pass
                # Continue to the next message, do not stream tool output as agent message
                continue

            # Stream agent messages
            words = content.split(' ')
            for word in words:
                yield {"source": source_name, "content": word + " "}

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        yield {"source": "System", "content": error_message}
    finally:
        await model_client.close()