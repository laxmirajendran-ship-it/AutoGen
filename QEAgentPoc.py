import asyncio
from pathlib import Path
import os
import json
import builtins
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, CodeExecutorAgent
from autogen_agentchat.conditions import TextMentionTermination,MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_core.tools import FunctionTool 
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage 
# from autogen_agentchat.agents import user_proxy_agent, assistant_agent   



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
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

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
        # tools=[get_functional_tool],
        input_func=custom_input,
        # tools=[],
        # system_message=
        #     "You are a TestManager. Take clear requirements from the user "
        #     "for creating UserStory and Acceptance Criteria."
        
    )
    # user_proxy_history_message = TestManager.on_messages(TestManager.name, cancellation_token=CancellationToken(),)

    # executor = CodeExecutorAgent(
    #     "executor",
    #     model_client=model_client,  # lets it narrate results
    #     code_executor=SimpleLocalCommandLineCodeExecutor(work_dir=Path.cwd() / "runs"),
    #     system_message=(
    #         "You are a skilled test case writer. Your task is to create detailed and effective test cases "
    #         "based on the requirements provided by the test_manager. Ensure that each test case includes clear steps, "
    #         "follows best practices and covers various scenarios including edge cases."
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
            ),
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
            "Once APPROVED is sent, Reply 'TERMINATE' to end the conversation."
        ),
    )



    
  
    
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
    max_messages_termination = MaxMessageTermination(max_messages=4)
    termination = text_mention_termination | max_messages_termination
    # termination = TextMentionTermination("exit", sources=["user"])
    # agent_team = RoundRobinGroupChat(
    #     # [user, coder, test_case_writer, test_case_reviewer], termination_condition=termination,
    #    [TestManager, test_case_writer, test_case_reviewer], termination_condition=termination,

    # # termination_condition=max_messages_termination,
    # )
    agent_team = SelectorGroupChat(
        # [user, coder, test_case_writer, test_case_reviewer], termination_condition=termination,
       [TestManager, test_case_writer, test_case_reviewer], termination_condition=termination,
       model_client=model_client,


    # termination_condition=max_messages_termination,
    )
    
        
    




    # user_proxy.run("I need a user story and acceptance criteria for a login feature.")
    # user_proxy.initiate_chat(group_chat_manager,"I need a user story and acceptance criteria for a login feature.")
    


    
    try:
        await Console(
            # user_proxy.run_stream("user",input_func="Provide your requirement (if not, give 'TERMINATE' to end the converstation)"),  # human in the loop
            # user_proxy.run_stream(),
            # agent_team.run_stream()
            agent_team.run_stream(task="Create and review test cases based on the requirements provided by the TestManager.")
            # agent_team.run_stream()
            # output_stats=True,
        )
    finally:
        await model_client.close()

        # Export the agent configuration to a JSON file
        # config = agent_team.dump_component()
        # print(config._dump_json())

if __name__ == "__main__":
    asyncio.run(main())