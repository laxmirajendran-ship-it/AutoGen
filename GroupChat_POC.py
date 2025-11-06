import asyncio

from pathlib import Path
import os
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen import ConversableAgent
from autogen_agentchat import ConversableAgent

# from autogen_agentchat import RetrieveUserProxyAgent
# from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
# from autogen_agentchat.agents import RetrieveUserProxyAgent

# alternate location in some package layouts
# from autogen_agentchat.agents.retrieve_user_proxy_agent import RetrieveUserProxyAgent
RetrieveUserProxyAgent = None
# OR, if the module lives in a submodule:
# from autogen_agentchat.agents.retrieve_user_proxy_agent import RetrieveUserProxyAgent


from autogen_agentchat.agents import RetrieveUserProxyAgent
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
# from autogen_agentchat.agents import GroupChat, GroupChatManager
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main() -> None:

    model_client = OpenAIChatCompletionClient(model="gpt-4.1-mini")

# def termination_msg(x):
#     return isinstance(x,dict) and "TERMINATE" == str(x.get("content",""))[-9:].upper()

    #CAll Test case writer agent
    test_manager = AssistantAgent(
        "test_manager",
        model_client=model_client,
        system_message = (
            "You are a Test Manager. Your task is to take clear requirements or update requests "
            "for creating test cases and pass them to test_case_writer. "
        ),

        # human_input_mode="Never",
        # max_consecutive_auto_replies=3,
        retrieve_config={
            "task":"code",
            "docs_path":"./docs",
            # "chunk_size":1000,
            "model":"gpt-4o-mini",
            "get_or_create": True, #True to use the existing collection
        },

        code_execution_config=False
        
    )

    #Test case writer agent
    test_case_writer = AssistantAgent(
        "test_case_writer",
        model_client=model_client,
        # is_termination_msg=termination_msg,
        # human_input_mode="NEVER",
        system_message=(
            "You are a skilled test case writer. Your task is to create detailed and effective test cases "
            "based on the requirements provided by the test_manager. Ensure that each test case includes clear steps, "
            "follows best practices and covers various scenarios including edge cases."
        ),
    )

    #Test case Reviewer agent
    test_case_reviewer = AssistantAgent(
        "test_case_reviewer",
        model_client=model_client,
        # is_termination_msg=termination_msg,
        # human_input_mode="NEVER",
        system_message=("You are a meticulous test case reviewer. Your task is to review the test cases created by the test_case_writer. "
            "Ensure that each test case is clear, comprehensive, and adheres to best practices. Provide constructive feedback "
            "and suggest improvements where necessary."
            "Approve only if all the review comments have been addressed by the test case writer agent"
            "Reply with either APPROVED or SUGGESTIONS for changes."
            "Once APPROVED is sent, Reply 'TERMINATE' to end the conversation."
        ),
    )

# groupchat = GroupChat(
#     agents=[test_case_writer, test_case_reviewer],
#     messsages = [],
#     max_rounds = 10,
#     speaker_selection_strategy="round_robin",)

# groupchat_manager = GroupChatManager(
#     # user_proxy_agent=test_manager,
#     group_chat=groupchat,
#     llm_config=llm_config,
#     human_input_mode="NEVER",
# )

    
    # user = UserProxyAgent("user")  # human in the loop

    termination = TextMentionTermination("TERMINATE"),
    agent_team = RoundRobinGroupChat(
        [test_manager, test_case_writer, test_case_reviewer], termination_condition=termination
    )

    try:
        await Console(
        agent_team.run_stream()
        )
    finally:
            await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())

# #Run the created agents
# def run_groupchat_poc():
#     test_manager.reset()
#     test_case_writer.reset()
#     test_case_reviewer.reset()  

#     # test_manager.initiate_chat(
#     #     groupchat_manager,
#     #     message=test_manager.message_generator,
#     #     problem="""   )
    
#     user = UserProxyAgent("user")  # human in the loop

#     termination = TextMentionTermination("exit", sources=["user"])
#     agent_team = RoundRobinGroupChat(
#         [user, coder, executor], termination_condition=termination
#     )
