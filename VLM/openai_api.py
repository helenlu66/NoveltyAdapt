import base64
from planning.TreeOfThoughtsPrompts import baseline_prompt
from utils import *
from langchain_openai import OpenAI
from langchain_core.messages import *
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents import AgentExecutor


from openai import OpenAI
client = OpenAI()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file = os.path.join(project_root, 'config.yaml')

config = load_config(config_file)

def chat_completion(prompt:str) -> str:
    """Generate a thought based on the given prompt.

    Args:
        prompt (str): prompt to the LLM
    """
    # gpt-o1 does not support setting the temperature parameter
    completion = client.chat.completions.create(
        model=config['vlm_agent']['model'],
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message.content

def code_completion(prompt:str) -> str:
    """Generate code based on the given prompt.

    Args:
        prompt (str): prompt to the LLM
    """
    completion = client.chat.completions.create(
        model=config['code_agent']['model'],
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message.content


state_evaluator = ChatOpenAI(model=config['vlm_agent']['model'], temperature=config['vlm_agent']['temperature'])

if __name__ == "__main__":
    # Example usage
    prompt = baseline_prompt
    response = chat_completion(prompt)
    print(f"Response: {response}")