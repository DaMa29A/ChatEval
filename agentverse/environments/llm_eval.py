# import asyncio
# import logging
# from typing import Any, Dict, List

# # from agentverse.agents.agent import Agent
# from agentverse.agents.conversation_agent import BaseAgent
# from agentverse.environments.rules.base import Rule
# from agentverse.message import Message

# from . import env_registry as EnvironmentRegistry
# from .basic import BasicEnvironment


# @EnvironmentRegistry.register("llm_eval")
# class LLMEvalEnvironment(BasicEnvironment):
#     """
#     An environment for prisoner dilema.
#     """

#     async def step(self) -> List[Message]:
#         """Run one step of the environment"""

#         # Get the next agent index
#         agent_ids = self.rule.get_next_agent_idx(self)

#         # Generate current environment description
#         env_descriptions = self.rule.get_env_description(self)

#         # Generate the next message
#         messages = await asyncio.gather(
#             *[self.agents[i].astep(self, env_descriptions[i]) for i in agent_ids]
#         )

#         # Some rules will select certain messages from all the messages
#         selected_messages = self.rule.select_message(self, messages)
#         self.last_messages = selected_messages
#         self.print_messages(selected_messages)

#         # Update the memory of the agents
#         self.rule.update_memory(self)

#         # Update the set of visible agents for each agent
#         self.rule.update_visible_agents(self)

#         self.cnt_turn += 1

#         return selected_messages



import asyncio
import logging
from typing import Any, Dict, List

# from agentverse.agents.agent import Agent
from agentverse.agents.conversation_agent import BaseAgent
from agentverse.environments.rules.base import Rule
from agentverse.message import Message

from . import env_registry as EnvironmentRegistry
from .basic import BasicEnvironment


@EnvironmentRegistry.register("llm_eval")
class LLMEvalEnvironment(BasicEnvironment):
    """
    An environment for llm_eval, based on BasicEnvironment.
    """

    async def step(self) -> List[Message]:
        """Run one step of the environment"""

        # === CORREZIONE DEFINITIVA: Utilizzo di self.__dict__ per aggirare Pydantic ===
        
        # 1. Accesso al contatore: Usiamo self.__dict__ per leggere/scrivere la variabile
        #    'agent_speak_counts'.
        speak_counts = self.__dict__.get('agent_speak_counts')
        
        if speak_counts is None:
            # Inizializza i contatori solo al primo step
            self.__dict__['agent_speak_counts'] = {agent.name: 0 for agent in self.agents}
            speak_counts = self.__dict__['agent_speak_counts']
            print("\n✅ Contatori Agenti inizializzati dinamicamente al primo step (usando __dict__).")
        # ==============================================================================

        # Get the next agent index
        agent_ids = self.rule.get_next_agent_idx(self)

        # === LOGICA AGGIUNTA PER IL CONTEGGIO E LA STAMPA ===
        
        # 2. Aggiorna il conteggio per gli agenti che sono stati selezionati per parlare
        current_agent_names = []
        for agent_id in agent_ids:
            agent_name = self.agents[agent_id].name
            speak_counts[agent_name] += 1
            current_agent_names.append(agent_name)
            
        # 3. Stampa lo stato attuale
        print(f"\n--- Inizio Turno di Discussione (Ciclo Ambiente): {self.cnt_turn + 1} ---")
        print("🗣️ Conteggio Turni di Parola per Agente:")
        
        for name, count in speak_counts.items():
            if name in current_agent_names:
                 # Evidenzia l'agente che parla in questo step
                 print(f"  -> {name}: {count} volte (PARLA ADESSO)")
            else:
                 print(f"  - {name}: {count} volte")
        print("---")
        # ====================================================

        # Generate current environment description
        env_descriptions = self.rule.get_env_description(self)

        # Generate the next message
        messages = await asyncio.gather(
            *[self.agents[i].astep(self, env_descriptions[i]) for i in agent_ids]
        )

        # Some rules will select certain messages from all the messages
        selected_messages = self.rule.select_message(self, messages)
        self.last_messages = selected_messages
        self.print_messages(selected_messages)

        # Update the memory of the agents
        self.rule.update_memory(self)

        # Update the set of visible agents for each agent
        self.rule.update_visible_agents(self)

        self.cnt_turn += 1

        return selected_messages