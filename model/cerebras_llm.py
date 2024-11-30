"""
pip install langchain-cerebras

os.environ["CEREBRAS_API_KEY"]
"""

import warnings
import os

from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any

from langchain_cerebras import ChatCerebras
from langchain_core._api.beta_decorator import LangChainBetaWarning
from langchain_core.rate_limiters import InMemoryRateLimiter

# InMemoryRateLimiter使用時の警告を消す
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from dotenv import load_dotenv

load_dotenv()  # .envの内容が環境変数として読み込まれる

# deplayment name
valid_model_names = Literal[
    "llama3.1-70b",
    "llama3.1-8b",
]


class CerebrasChatBase(ChatCerebras):
    def __init__(
            self,
            model_name: Literal[valid_model_names],
            requests_per_second: Optional[float] = None,
            **kwargs,
    ):
        api_key = os.getenv("CEREBRAS_API_KEY")
        shared_kwargs = dict(
            api_key=api_key,
            model=model_name,
            **kwargs,
        )

        if requests_per_second:
            r_limiter = InMemoryRateLimiter(requests_per_second=requests_per_second)
            shared_kwargs["rate_limiter"] = r_limiter

        super().__init__(**shared_kwargs)


if __name__ == "__main__":
    """
    python -m model.cerebras_llm
    """

    llm = CerebrasChatBase(
        model_name="llama3.1-70b",
        requests_per_second=0.32,
    )

    res = llm.invoke("hello")
    print(res)


