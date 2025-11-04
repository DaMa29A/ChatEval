from __future__ import annotations

import re
from typing import Union

from agentverse.parser import OutputParser, LLMResult

from agentverse.utils import AgentAction, AgentFinish

from agentverse.parser import OutputParserError, output_parser_registry


@output_parser_registry.register("llmeval")
class LLMEvalParser(OutputParser):
    def parse(self, output: LLMResult, cnt_turn: int, max_turns: int, agent_nums: int) -> Union[AgentAction, AgentFinish]:
        # TODO: modifiche a parser
        # Formato output: 
        '''
        content='Evaluation evidence: The assistant’s response is brief and somewhat relevant, as it acknowledges the user’s 
        reluctance to share details about the meeting and attempts to connect by suggesting the meeting might be boring. 
        However, it lacks empathy and does not encourage further conversation or provide any supportive or engaging content. 
        The response could have been improved by showing understanding of the user’s privacy or by offering a light, 
        friendly comment to keep the dialogue flowing. Overall, it is a minimal but acceptable reply that fits the context 
        but does not add much value.\n\nOverall Score: 5' 
        send_tokens=626 
        recv_tokens=111 
        total_tokens=737
        '''
        text = output.content
        #print(f"text:\n{text}")
        cleaned_output = text.strip() # rimozione spazi bianchi, tab, newline a inizio e fine stringa
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output) # sostituisce + newline con uno solo
        cleaned_output = cleaned_output.split("\n") # ad ogni \n crea una stringa, quindi ora cleaned_output è lista di stringhe
        #print(f"cleaned output:\n{cleaned_output}")

        # Abbiamo eliminato il controllo
        # if cnt_turn >= max_turns - agent_nums:
        #     # if not cleaned_output[0].startswith("Answer") :
        #     if not (cleaned_output[-2].startswith("The score of Assistant 1:") and \
        #             cleaned_output[-1].startswith("The score of Assistant 2:")):
        #         raise OutputParserError(text)

        return AgentFinish({"output": text}, text)

@output_parser_registry.register("fed")
class FedParser(OutputParser):
    def parse(self, output: LLMResult, cnt_turn: int, max_turns: int, agent_nums: int) -> Union[AgentAction, AgentFinish]:
        pass
