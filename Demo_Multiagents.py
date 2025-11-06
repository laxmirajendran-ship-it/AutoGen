import asyncio
from pathlib import Path
from behave import *
import os
import json
import builtins
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, CodeExecutorAgent
from autogen_agentchat.conditions import TextMentionTermination,MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient    
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_core.tools import FunctionTool 
from datetime import datetime
 

#Python function to write output of the agents to a file and save it in the outputs folder.
def write_file(filename: str, content: str) -> dict:
    print(filename, content)
    work_dir = Path.cwd() / "outputs"
    work_dir.mkdir(exist_ok=True)
    p = work_dir / filename
    p.write_text(content, encoding="utf-8")
    # print("Saved CSV to:", p)
    return {"path": str(p)}

write_file_tool = FunctionTool(
    write_file,
    name="write_file",
    description="Write given content to a file in the current working directory. Args: filename, content",
)

#Custom input function to get user requirements to overright the text of default prompt
def custom_input(prompt: str) -> str:
    return builtins.input("Enter your requirements (or type 'TERMINATE' to end): ")

# add a concrete subclass that implements the abstract lifecycle methods
class SimpleLocalCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
    async def start(self):
        # no-op startup; implement real startup logic if needed
        return None

    async def stop(self):
        # no-op shutdown; implement real shutdown logic if needed
        return None


#Multiple agents working together to create test cases based on user requirements.
async def main() -> None:

    #gpt-5-nano gpt-4o-mini gpt-5
    # model_client = OpenAIChatCompletionClient(model="gpt-4.1", temperature=0)
    model_client = OllamaChatCompletionClient(model="llama3", temperature=0)
    model_context = BufferedChatCompletionContext(buffer_size=5)

    # Agent to take user requirements and create user story and acceptance criteria in gherkin syntax.
    TestManager = UserProxyAgent(
        "TestManager",
        input_func=custom_input,
        description=
            "You are a TestManager. Take clear requirements from the user only once"
            "Create UserStory and Acceptance Criteria from the given requirements"
            "Userstory and Acceptance Criteria to be written in a well-structured Gherkin syntax."
            "Ensure you use the correct Gherkin keywords: Feature, Scenario, Given, When, Then, And, But. "
            "get input from the user to terminate the conversation by typing 'TERMINATE'.",
    )

    # Agent that creates test cases in csv format based on the user story and acceptance criteria provided by the TestManager.  
    test_case_writer = AssistantAgent(
            "test_case_writer",
            model_client=model_client,  # lets it narrate results
            # code_executor=SimpleLocalCommandLineCodeExecutor(work_dir=Path.cwd() / "runs"),
            system_message=(
                "You are a skilled test case writer. Your task is to create detailed and effective test cases in csv format "
                "based on the requirements provided by the test_manager. Ensure that each test case includes clear steps, "
                "follows best practices as mentioned in the documents in ./docs folder."
                "Produce test cases in csv format with the following columns in the exact order: "
                "1. Test Case ID (Unique identifier, e.g., TC_001)"
                "2. Test Case Name (Brief, descriptive title)"
                "3. Preconditions (Any setup required before executing the test)"
                "4. Test Steps (Numbered steps to execute the test)"
                "5. Expected Result (The expected outcome if the test passes)"
                "Ensure each column is properly populated for each test case"
                "When you have completed writing all the test cases, save it in a csv file and "
                "name the csv file in the format 'Test_Cases_MMDDYYY_HHMMSS.csv'"),                
                    
                # "When finished, CALL the tool 'write_file' with two arguments: "
                # tools=[write_file_tool],

        )


    #Test case Reviewer agent
    test_case_reviewer = AssistantAgent(
        "test_case_reviewer",
        model_client=model_client,
        # is_termination_msg=termination_msg,
        # human_input_mode="NEVER",
        system_message=("You are a meticulous test case reviewer. Your task is to review the test cases created by the test_case_writer. "
                "follows best practices as mentioned in the documents in ./docs folder."
            # "Ensure that each test case is clear, comprehensive, and adheres to best practices."
            "Provide constructive feedback "
            "and suggest improvements where necessary."
            "Approve only if all the review comments have been addressed by the test case writer agent"
            "Reply with either APPROVED or SUGGESTIONS for changes."
        ),
    )

    bdd_coder = AssistantAgent(
        "bdd_coder",
        model_client=model_client,

        system_message=("You are an expert BDD Coder who writes step definitions in Python using the 'behave' library syntax. "
                   "You must take the input Feature file content provided by the TestManager and generate corresponding step definition code. "
                   "Ensure all 'Given', 'When', 'And', 'But' and 'Then' steps are covered with basic 'pass' implementations. "
                   "Make sure to import necessary modules from 'behave' and structure the code correctly."
                   "Save the step definitions in a Python file named 'steps_<FeatureName>.py', where <FeatureName> is derived from the Feature title."
                   "Save the Python file in the current working directory under output folder with name 'steps_<FeatureName>_MMDDYYYY_HHMMSS.py'. "
                   "Reply with 'TERMINATE' once the Python file content is fully written and correct."),

        # tools=[write_file_tool],    

    )
    
        
    text_mention_termination = TextMentionTermination("TERMINATE")
    termination = text_mention_termination

    #Create a team of agents to work together to complete the task.
    agent_team = RoundRobinGroupChat(

       [TestManager, test_case_writer, test_case_reviewer, bdd_coder], termination_condition=termination,
    #    [TestManager,bdd_coder], termination_condition=termination,
       max_turns = 4
    )
    
    
    try:
        await Console(
           
            agent_team.run_stream(task="Create and review test cases based on the requirements provided by the TestManager.")
        )
    finally:
        await model_client.close()

        # # Export the agent configuration to a JSON file
        # config = agent_team.dump_component()
        # print(config.model_dump_json())

if __name__ == "__main__":
    asyncio.run(main())