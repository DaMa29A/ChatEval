import os
# os.environ["OPENAI_API_KEY"] = "***"
# os.environ["OPENAI_BASE_URL"] = "***"

# always remember to put these lines at the top of your code if you are using clash
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ["all_proxy"] = "socks5://127.0.0.1:7890"


import json
from eval_helper.get_evaluation import get_evaluation

from agentverse.agentverse import AgentVerse
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--reverse_input", default=False, action="store_true")


args = parser.parse_args()

agentverse, args_data_path, args_output_dir = AgentVerse.from_task(args.config)

print(args)

os.makedirs(args_output_dir, exist_ok=True)
with open(os.path.join(args_output_dir, "args.txt"), "w") as f:
    f.writelines(str(args))

# uncomment this line if you don't want to overwrite your output_dir
# if os.path.exists(args_output_dir) and len(os.listdir(args_output_dir)) > 1 :
#
#     raise ValueError("the output_dir is not empty, check if is expected.")

with open(args_data_path) as f:
    data = json.load(f)

if "faireval" in args_data_path:
    pair_comparison_output = []

    for num, ins in enumerate(data[:80]):

        print(f"================================instance {num}====================================")

        # reassign the text to agents, and set final_prompt to null for debate at first round
        for agent_id in range(len(agentverse.agents)):
            agentverse.agents[agent_id].source_text = ins["question"]

            if args.reverse_input:
                agentverse.agents[agent_id].compared_text_one = ins["response"]["vicuna"]
                agentverse.agents[agent_id].compared_text_two = ins["response"]["gpt35"]
            else:
                agentverse.agents[agent_id].compared_text_one = ins["response"]["gpt35"]
                agentverse.agents[agent_id].compared_text_two = ins["response"]["vicuna"]

            agentverse.agents[agent_id].final_prompt = ""

        agentverse.run()

        evaluation = get_evaluation(setting="every_agent", messages=agentverse.agents[0].memory.messages, agent_nums=len(agentverse.agents))

        pair_comparison_output.append({"question": ins["question"],
                                       "response": {"gpt35": ins["response"]["gpt35"],
                                                    "vicuna": ins["response"]["vicuna"]},
                                       "evaluation": evaluation})

        os.makedirs(args_output_dir, exist_ok=True)
        with open(os.path.join(args_output_dir, "pair_comparison_results.json"), "w") as f:
            json.dump(pair_comparison_output, f, indent=4)
    # with open(os.path.join(args_output_dir, "gt_origin_results.json"), "w") as f:
    #     json.dump(gt_origin_output, f, indent=4)

elif "fed" in args_data_path:
    print(f"--- Rilevato dataset 'Fed'. Avvio modalità di valutazione assoluta. ---")
    output = []
    
    # n=1 significa che stai testando solo sul primo elemento.
    # Ricorda di rimuovere [:n] per l'esecuzione finale!
    n = 1 
    
    # 1. LOOP SUL DATASET (Esterno)
    for index, elem in enumerate(data[:n]):
        print(f"================================instance {index+1}====================================")
        
        chat = elem["context"]
        response = elem["response"]
        #print(f"Chat:\n{chat}\n\nResponse:\n{response}\n")

        for agent_id in range(len(agentverse.agents)):
            agent = agentverse.agents[agent_id]
            #print(f"Agente '{agent.name}'")
            
            # Assegna gli stessi dati a ogni agente
            agent.source_text = chat
            agent.response_to_evaluate = response
            agent.final_prompt = "" # Resetta i prompt per il dibattito

        print("--- Dati agenti impostati. Avvio del dibattito (agentverse.run())... ---")
        # Vedendo nella funzione astep, stampando response ne vediamo il formato
        # Nel nostro caso abbiamo qualcosa di simile:
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
        agentverse.run() #avvio dibattito

        # Estrazione risultati dal dibattito
        print("--- Dibattito concluso. Estrazione delle valutazioni... ---")
        evaluation = get_evaluation(setting="every_agent", messages=agentverse.agents[0].memory.messages, agent_nums=len(agentverse.agents))
        #print(f"Evaluation: {evaluation}")
        # 5. SALVA L'OUTPUT (UNA VOLTA SOLA)
    #     output.append({
    #         "context": elem["context"],
    #         "response": elem["response"],
    #         "human_annotations": elem["annotations"], # Ci salviamo anche i voti umani
    #         "chateval_evaluation": evaluation
    #     })

    # # 6. SALVA IL FILE JSON FINALE (Alla fine di tutto il ciclo)
    # os.makedirs(args_output_dir, exist_ok=True)
    # with open(os.path.join(args_output_dir, "fed_evaluation_results.json"), "w") as f:
    #     print(f"--- Valutazione completata. Salvo i risultati in {args_output_dir}/fed_evaluation_results.json ---")
    #     json.dump(output, f, indent=4)

elif "tropical" in args_data_path:
    print(f"Tropical")
    output = []
    n = 1
    for index, elem in enumerate(data[:n]):
        print(f"================================instance {index+1}====================================")
