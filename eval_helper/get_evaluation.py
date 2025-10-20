import os
import json
from typing import List
from agentverse.message import Message


def get_evaluation(setting: str = None, messages: List[Message] = None, agent_nums: int = None) -> List[dict]:
    #TODO: Modifica effettuata anche qui
    results = []
    if setting == "every_agent":
        # Currently 2 round, concurrent, so the response will start from messages[-3:]
        for message in messages[-agent_nums:]:
            #print(f"Mex: {message}")
            print(f"\n--Role: {message.sender}\n--Evaluation:\n{message.content}")
            results.append({"role": message.sender,
                            "evaluation": message.content})

    return results
