    # Example: my_autogen_app.py
# from autogen import AssistantAgent, UserProxyAgent
import os
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
# from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel
# from autogen_agentchat.code_executors.local import LocalCommandLineCodeExecutor

# from autogen import config_list_from_json
# config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")

    # Load configuration from a JSON file or environment variables
# config_list = [
#     {
   
#         "model": ["gpt-4", "gpt-3.5-turbo","llama3:instruct"],
#         'api_key': os.environ.get("OPENAI_API_KEY"),
#     },
# ]

model_client = OpenAIChatCompletionClient(
    model="gpt-4",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# llm_config = {

#     "request_timeout": 600,
#     "seed": 42,    
#     "temperature": 0,
# }

# executor = LocalCommandLineCodeExecutor(work_dir="web")
# Create an assistant agent
assistant = AssistantAgent(name="assistant",model_client=model_client)

# Set a working directory
working_dir = "web"
os.makedirs(working_dir, exist_ok=True)
# Create a user proxy agent with code execution capabilities

user_proxy = UserProxyAgent(name="user_proxy",
                            human_input_mode="TERMINATE",
    # code_execution_config={"work_dir": "coding", "use_docker": False}, # Set use_docker to True for sandboxed execution
    code_execution_config={"work_dir":"web","use_docker": False},
    # code_execution_config=False,
    # code_execution_config={"executor": executor},
    # max_consecutive_auto_reply=4,
     # Or "ALWAYS", "TERMINATE"
    system_message="""Reply TERMINATE if the task has been solved at full statisfaction. Otherwise, reply ALWAYS to continue the task.""",
)

task = """ write python code to find prime numbers between 1 and 100 and save the output in a text file named 'primes.txt'."""

# Initiate a chat
user_proxy.initiate_chat(
    assistant,
    message=task,
)