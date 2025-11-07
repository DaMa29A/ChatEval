import os
import json
import re
from typing import List
from agentverse.message import Message


def get_evaluation(setting: str = None, messages: List[Message] = None, agent_nums: int = None, type: str = None) -> List[dict]:
    #TODO: Modifica effettuata anche qui
    results = []
    if setting == "every_agent":
        print(f"Extracting evaluation for {agent_nums} agents.")
        print(f"Total messages: {len(messages)}")
        print(f"Evaluation extraction type: {type}")  
        for message in messages[-agent_nums:]:
            mex = message
            agent_role = mex.sender
            evaluation = mex.content
            print(f"Raw evaluation from {agent_role}: {evaluation}")
            if type == "fed":
                # 1. (.*)   - Gruppo 1: Cattura tutto il testo dell'analisi
                # 2. (\d+)  - Gruppo 2: Cattura solo i numeri (lo score)
                match = re.search(r"(.*)Overall Score:\s*(\d+)", evaluation, flags=re.DOTALL | re.IGNORECASE)
                score = None
                evaluation_text = evaluation

                if match:
                    evaluation_text = match.group(1).strip()
                    score = int(match.group(2))
                results.append({"role": agent_role,
                                "evaluation": evaluation_text,
                                "score": score})
                
            elif type == "topical":
                pass
            else:
                results.append({"role": agent_role,
                                "evaluation": evaluation})

    return results
