import logging
import numpy as np
import time
import os
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from agentverse.llms.base import LLMResult
from . import llm_registry
from .base import BaseChatModel, BaseCompletionModel, BaseModelArgs
from agentverse.message import Message
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

MODEL_NAME = os.getenv('GROQ_MODEL_NAME')   # Viene registrato giù
#MODEL_NAME = os.getenv('FREE_GPT_MODEL_NAME')

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    
    # TODO: originale
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
    # aclient = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
    
    # Free GPT-4
    # OPENAI_API_KEY = os.getenv('FREE_GPT_KEY')
    # OPENAI_BASE_URL = os.getenv('FREE_GPT_BASE_URL')
    # print(f"Print key: {OPENAI_API_KEY}")
    # print(f"Base url: {OPENAI_BASE_URL}")
    # print(f"Model name: {MODEL_NAME}")
    # client = OpenAI(api_key=OPENAI_API_KEY, base_url= OPENAI_BASE_URL)
    # aclient = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    #Groq llama
    OPENAI_API_KEY = os.getenv('GROQ_KEY')
    OPENAI_BASE_URL = os.getenv('GROQ_BASE_URL')
    
    print(f"Print key: {OPENAI_API_KEY}")
    print(f"Base url: {OPENAI_BASE_URL}")
    print(f"Model name: {MODEL_NAME}")
    client = OpenAI(api_key=OPENAI_API_KEY, base_url= OPENAI_BASE_URL)
    aclient = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


    from openai import OpenAIError
except ImportError:
    is_openai_available = False
    logging.warning("openai package is not installed")
else:
    if openai.api_key is None:
        logging.warning(
            "OpenAI API key is not set. Please set the environment variable OPENAI_API_KEY"
        )
        is_openai_available = False
    else:
        is_openai_available = True


class OpenAIChatArgs(BaseModelArgs):
    model: str = Field(default=MODEL_NAME)
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=1.0)
    top_p: int = Field(default=1)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)


class OpenAICompletionArgs(OpenAIChatArgs):
    #model: str = Field(default="text-davinci-003")
    model: str = Field(default=MODEL_NAME)
    suffix: str = Field(default="")
    best_of: int = Field(default=1)


# @llm_registry.register("text-davinci-003")
# @llm_registry.register(MODEL_NAME)
# class OpenAICompletion(BaseCompletionModel):
#     args: OpenAICompletionArgs = Field(default_factory=OpenAICompletionArgs)
#     print("OpenAICompletion SONO QUI")

#     def __init__(self, max_retry: int = 3, **kwargs):
#         args = OpenAICompletionArgs()
#         args = args.dict()
#         for k, v in args.items():
#             args[k] = kwargs.pop(k, v)
#         if len(kwargs) > 0:
#             logging.warning(f"Unused arguments: {kwargs}")
#         super().__init__(args=args, max_retry=max_retry)

#     def generate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
#         print(f"Client: {client}")
#         response = client.completions.create(prompt=prompt, **self.args.dict())
#         return LLMResult(
#             content=response.choices[0].text,
#             send_tokens=response.usage.prompt_tokens,
#             recv_tokens=response.usage.completion_tokens,
#             total_tokens=response.usage.total_tokens,
#         )

#     async def agenerate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
#         response = await aclient.completions.create(prompt=prompt, **self.args.dict())
#         return LLMResult(
#             content=response.choices[0].text,
#             send_tokens=response.usage.prompt_tokens,
#             recv_tokens=response.usage.completion_tokens,
#             total_tokens=response.usage.total_tokens,
#         )

@llm_registry.register("text-davinci-003")
@llm_registry.register(MODEL_NAME)
class OpenAICompletion(BaseCompletionModel):
    args: OpenAICompletionArgs = Field(default_factory=OpenAICompletionArgs)

    # Nuovo metodo helper per pulire gli argomenti prima della chiamata Chat
    def _prepare_chat_args(self) -> dict:
        chat_args = self.args.dict()
        # Questi argomenti non sono supportati dalla Chat API, vanno rimossi
        chat_args.pop('suffix', None) 
        chat_args.pop('best_of', None)
        return chat_args

    # Metodo helper per convertire il prompt di completion in formato message di chat
    def _to_chat_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]
        
    def generate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        messages = self._to_chat_messages(prompt)
        chat_args = self._prepare_chat_args() # Usa gli argomenti puliti

        try:
            # Chiama l'endpoint CHAT con gli argomenti puliti (FIX 404 e FIX suffix)
            response = client.chat.completions.create(
                messages=messages, 
                **chat_args
            )
        except (OpenAIError, KeyboardInterrupt) as error:
            raise
        return LLMResult(
            content=response.choices[0].message.content, 
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    async def agenerate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        messages = self._to_chat_messages(prompt)
        chat_args = self._prepare_chat_args() # Usa gli argomenti puliti

        try:
            # Chiama l'endpoint CHAT con gli argomenti puliti (FIX 404 e FIX suffix)
            response = await aclient.chat.completions.create(
                messages=messages, 
                **chat_args
            )
        except (OpenAIError, KeyboardInterrupt) as error:
            raise
        return LLMResult(
            content=response.choices[0].message.content,
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )


@llm_registry.register(MODEL_NAME)
@llm_registry.register("gpt-3.5-turbo-0301")
@llm_registry.register("gpt-3.5-turbo")
@llm_registry.register("gpt-4")
class OpenAIChat(BaseChatModel):
    args: OpenAIChatArgs = Field(default_factory=OpenAIChatArgs)

    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAIChatArgs()
        args = args.dict()

        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logging.warning(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)

    def _construct_messages(self, prompt: str, chat_memory: List[Message], final_prompt: str):
        chat_messages = []
        for item_memory in chat_memory:
            chat_messages.append(str(item_memory.sender) + ": " + str(item_memory.content))
        processed_prompt = [{"role": "user", "content": prompt}]
        for chat_message in chat_messages:
            processed_prompt.append({"role": "assistant", "content": chat_message})
        processed_prompt.append({"role": "user", "content": final_prompt})
        return processed_prompt

    def generate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt, chat_memory, final_prompt)
        try:
            if openai.api_type == "azure":
                response = client.chat.completions.create(engine="gpt-4-6", messages=messages, **self.args.dict())
            else:
                response = client.chat.completions.create(messages=messages, **self.args.dict())
        except (OpenAIError, KeyboardInterrupt) as error:
            raise
        return LLMResult(
            content=response.choices[0].message.content,
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    async def agenerate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt, chat_memory, final_prompt)
        try:
            if openai.api_type == "azure":
                response = await aclient.chat.completions.create(engine="gpt-4-6", messages=messages, **self.args.dict())
            else:

                response = await aclient.chat.completions.create(messages=messages, **self.args.dict())
        except (OpenAIError, KeyboardInterrupt) as error:
            raise
        return LLMResult(
            content=response.choices[0].message.content,
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )


def get_embedding(text: str, attempts=3) -> np.array:
    attempt = 0
    while attempt < attempts:
        try:
            text = text.replace("\n", " ")
            embedding = client.embeddings.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]
            return tuple(embedding)
        except Exception as e:
            attempt += 1
            logger.error(f"Error {e} when requesting openai models. Retrying")
            time.sleep(10)
    logger.warning(
        f"get_embedding() failed after {attempts} attempts. returning empty response"
    )