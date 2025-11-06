import asyncio
from pathlib import Path
from behave import *
import os
print("Current working directory:", os.getcwd())
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
# from autogen_agentchat.agents import user_proxy_agent, assistant_agent   


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


def custom_input(prompt: str) -> str:
    # The default prompt from the base class is often ignored in favor
    # of the hardcoded one in the base implementation. 
    # We explicitly use our custom prompt here.
    return builtins.input("Enter your requirements (or type 'TERMINATE' to end): ")
        
#         # You can add logic to print the original prompt content if needed
#         # print(f"Original prompt content: {prompt}") 

#         return input(custom_prompt)

# add a concrete subclass that implements the abstract lifecycle methods
class SimpleLocalCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
    async def start(self):
        # no-op startup; implement real startup logic if needed
        return None

    async def stop(self):
        # no-op shutdown; implement real shutdown logic if needed
        return None

async def main() -> None:

    
       # 2. Wrap the function with FunctionTool.
    # get_functional_tool = FunctionTool(
    # main,
    # description="Tool to get User requirements.",
# )
#gpt-5-nano gpt-4o-mini
    model_client = OpenAIChatCompletionClient(model="gpt-5-nano", temperature=1)
    # model_client = OllamaChatCompletionClient(model="llama3", temperature=0)
    model_context = BufferedChatCompletionContext(buffer_size=5)

     # Define your user proxy agent (human in the loop)
    # user_proxy_agent.system_message = "My custom message."
    # user_proxy = user_proxy_agent(
    #     name="UserProxy",
        
    #     # human_input_mode="ALWAYS", # Or "NEVER", "AUTO"
    #     # code_execution_config={"work_dir": "coding", "use_docker": False}
    # )

    TestManager = UserProxyAgent(
        "TestManager",
        # model_client=model_client,
        # # tools=[get_functional_tool],
        # tools=[],
        input_func=custom_input,
        description=
            "You are a TestManager. Take clear requirements from the user only once"
            "Create UserStory and Acceptance Criteria from the given requirements"
            "Userstory and Acceptance Criteria to be written in a well-structured Gherkin syntax."
            "Ensure you use the correct Gherkin keywords: Feature, Scenario, Given, When, Then, And, But. "
            "get input from the user to terminate the conversation by typing 'TERMINATE'.",
    )

    # executor = CodeExecutorAgent(
    #     "executor",
    #     model_client=model_client,  # lets it narrate results
    #     code_executor=SimpleLocalCommandLineCodeExecutor(work_dir=Path.cwd() / "outputs"),
    #     system_message=(
    #         "Your role is to get the generated test cases from the test_case_writer agent and upload them to the system in the given path"
    #     ),
    # )

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

                #  "return the output as a Markdown table"
                # "Convert the Markdown table to csv format."
                # # "Use comma as the delimiter and enclose text fields in double quotes where necessary."
                # "Do not include any additional commentary or explanation — provide ONLY the csv content."
                
                    
                # "When finished, CALL the tool 'write_file' with two arguments: "
                tools=[write_file_tool],
                # "filename (e.g. 'Test_Cases_Password_Reset_YYYYMMDD_HHMMSS.csv') and content (the CSV text). "
                # "Do not only print CSV to chat — invoke the tool so the file is saved automatically."

        )

    # content = await test_case_writer._model_context.get_messages()

    # #Agent that converts csv file to excel file.
    # csv_converter = AssistantAgent(
    #         "csv_converter",
    #         model_client=model_client,  # lets it narrate results
    #         # code_executor=SimpleLocalCommandLineCodeExecutor(work_dir=Path.cwd() / "runs"),
    #         system_message=("You are an expert CSV to Excel converter. Your task is to convert the given CSV content into an Excel file format. "),
    #  )

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
        # is_termination_msg=termination_msg,
        # human_input_mode="NEVER",
        system_message=("You are an expert BDD Coder who writes step definitions in Python using the 'behave' library syntax. "
                   "You must take the input Feature file content provided by the TestManager and generate corresponding step definition code. "
                   "Ensure all 'Given', 'When', 'And', 'But' and 'Then' steps are covered with basic 'pass' implementations. "
                   "Make sure to import necessary modules from 'behave' and structure the code correctly."
                   "Save the step definitions in a Python file named 'steps_<FeatureName>.py', where <FeatureName> is derived from the Feature title."
                   "Save the Python file in the current working directory under output folder with name 'steps_<FeatureName>_MMDDYYYY_HHMMSS.py'. "
                   "Reply with 'TERMINATE' once the Python file content is fully written and correct."),

        tools=[write_file_tool],
    

    )
    


    # user_proxy = UserProxyAgent(
    #     name= "user_proxy",
    #     input_func=custom_input,
    #     )
  
    
    # group_chat = GroupChat(
    #     [user_proxy, coder, executor],
    # )

    # # Initiate the chat
    # group_chat_manager = GroupChatManager(
    #     group_chat=group_chat,
    # # The user_proxy is a part of the group chat and will also be the one initiating the conversation.
    # )
    
    # coder.initiate_chat(user_proxy, message="Provide your requirement to get user story and acceptance criteria.")
    text_mention_termination = TextMentionTermination("TERMINATE")
    # max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination
    # termination = TextMentionTermination("exit", sources=["user"])
    # agent_team = RoundRobinGroupChat(
    #     # [user, coder, test_case_writer, test_case_reviewer], termination_condition=termination,
    #      [user_proxy, test_case_writer, test_case_reviewer], termination_condition=termination,
    # #     # messages=[],
    #     max_turns=5
    # )

    agent_team = RoundRobinGroupChat(
        # [user, coder, test_case_writer, test_case_reviewer], termination_condition=termination,
       [TestManager, test_case_writer, test_case_reviewer, bdd_coder], termination_condition=termination,
    #    [TestManager, test_case_writer], termination_condition=termination,
    #    model_client=model_client,
    #    model_context=model_context,
       max_turns = 4,

    # termination_condition=max_messages_termination,
    )
    
    # user_proxy.run("I need a user story and acceptance criteria for a login feature.")
    # user_proxy.initiate_chat(group_chat_manager,"I need a user story and acceptance criteria for a login feature.")
    
    try:
        await Console(
            # user_proxy.run_stream("user",input_func="Provide your requirement (if not, give 'TERMINATE' to end the converstation)"),  # human in the loop
            agent_team.run_stream(task="Create and review test cases based on the requirements provided by the TestManager.")
        )
    finally:
        await model_client.close()

        # Export the agent configuration to a JSON file
        config = agent_team.dump_component()
        print(config.model_dump_json())

if __name__ == "__main__":
    asyncio.run(main())