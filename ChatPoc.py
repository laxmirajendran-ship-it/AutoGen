import asyncio
from pathlib import Path
import os
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4.1-mini")

    coder = AssistantAgent(
        "coder",
        model_client=model_client,
        system_message=(
            "You are a Product owner. Take clear requirements from the user "
            "for creating UserStory and Acceptance Criteria."
        ),
    )

    executor = CodeExecutorAgent(
        "executor",
        model_client=model_client,  # lets it narrate results
        code_executor=LocalCommandLineCodeExecutor(work_dir=Path.cwd() / "runs"),
    )

    user = UserProxyAgent("user")  # human in the loop

    termination = TextMentionTermination("exit", sources=["user"])
    team = RoundRobinGroupChat(
        [user, coder, executor], termination_condition=termination
    )

    try:
        await Console(
            team.run_stream()
        )
    finally:
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())