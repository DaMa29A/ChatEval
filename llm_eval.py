import os
import json
from eval_helper.get_evaluation import get_evaluation
from agentverse.agentverse import AgentVerse
from argparse import ArgumentParser
import time

parser = ArgumentParser()

parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--reverse_input", default=False, action="store_true")

args = parser.parse_args()

agentverse, args_data_path, args_output_dir = AgentVerse.from_task(args.config)

print(args)

os.makedirs(args_output_dir, exist_ok=True)
with open(os.path.join(args_output_dir, "args.txt"), "w") as f:
    f.writelines(str(args))

with open(args_data_path, 'r', encoding='utf-8') as f:
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
    print(f"[Rilevato dataset 'Fed']")
    output = []
    
    # n=1 significa che stai testando solo sul primo elemento.
    # Ricorda di rimuovere [:n] per l'esecuzione finale!
    n = 10
    
    # 1. LOOP SUL DATASET (Esterno)
    #TODO : One-to-One evaluation fai data[50:80] per eseguire su 30 istanze diverse
    #TODO : Simultaneous evaluation fai data[50:80] per eseguire su 30 istanze diverse
    for index, elem in enumerate(data[:1]): #data[:n]   data[x:y]
        print(f"================================instance {index+1}====================================")
        
        chat = elem["context"]
        response = elem["response"]
        #print(f"Chat:\n{chat}\n\nResponse:\n{response}\n")

        for agent_id in range(len(agentverse.agents)):
            agent = agentverse.agents[agent_id]
            
            # Assegna gli stessi dati a ogni agente
            agent.source_text = chat
            agent.response_to_evaluate = response
            agent.final_prompt = "" # Resetta i prompt per il dibattito

        print("--- Dati agenti impostati. Avvio del dibattito (agentverse.run())... ---")
        # Vedendo nella funzione astep, stampando response ne vediamo il formato
        
        # LOG 1: VERIFICA LA MEMORIA PRIMA
        print(f"--- DEBUG [llm_eval]: Pre-run. Messaggi in memoria: {len(agentverse.agents[0].memory.messages)} ---")
        agentverse.run() #avvio dibattito
        # LOG 2: VERIFICA LA MEMORIA SUBITO DOPO
        full_transcript = agentverse.agents[0].memory.messages
        print(f"--- DEBUG [llm_eval]: Post-run. Messaggi in memoria: {len(full_transcript)} ---")
        if len(full_transcript) == 0:
            print("--- DEBUG [llm_eval]: ERRORE? La memoria è vuota dopo agentverse.run()! ---")

        # --- INIZIO BLOCCO SALVATAGGIO TRANSCRIZIONE TXT ---
        
        # Poiché la visibilità è "all", tutti gli agenti hanno la stessa cronologia.
        # Accediamo alla memoria del primo agente (agente 0).
        full_transcript = agentverse.agents[0].memory.messages
        
        # Definisci dove salvare il file .txt (usa lo stesso args_output_dir del JSON)
        # Assicurati che 'os' sia importato all'inizio del tuo script (import os)
        transcript_filename = os.path.join(args_output_dir, f"istanza_{index+1}_transcript.txt")

        try:
            with open(transcript_filename, "w", encoding="utf-8") as f:
                f.write(f"--- TRANSCRIZIONE DIBATTITO (Istanza {index+1}) ---\n\n")
                f.write(f"CONTEXT:\n{chat}\n\n")
                f.write(f"RESPONSE:\n{response}\n\n")
                f.write("--- INIZIO DIBATTITO ---\n\n")
                
                for message in full_transcript:
                    # Scrivi il mittente (es. "Critic") e il contenuto del messaggio nel file
                    f.write(f"[{message.sender}]: {message.content}\n\n") # Aggiungo due ritorni a capo per leggibilità
            
            print(f"--- Trascrizione salvata in: {transcript_filename} ---")

        except Exception as e:
            print(f"ERRORE durante il salvataggio della trascrizione: {e}")

        # --- FINE BLOCCO SALVATAGGIO TRANSCRIZIONE TXT ---
        
        # LOG 3: VERIFICA I MESSAGGI PASSATI A GET_EVALUATION
        print(f"--- DEBUG [llm_eval]: Passo {len(full_transcript)} messaggi a get_evaluation ---")
        evaluation = get_evaluation(setting="every_agent", messages=full_transcript, agent_nums=len(agentverse.agents), type="fed")
        print(f"Evaluation: {evaluation}")

        # Estrazione risultati dal dibattito
        print("--- Dibattito concluso. Estrazione delle valutazioni... ---")
        evaluation = get_evaluation(setting="every_agent", messages=agentverse.agents[0].memory.messages, agent_nums=len(agentverse.agents), type="fed")
        print(f"Evaluation: {evaluation}")

        annotations = elem["annotations"]
        overall_annotations = annotations.get("Overall")
        average_scores = 0
        if overall_annotations:
            sum_scores = sum(overall_annotations)
            num_scores = len(overall_annotations)
            average_scores = sum_scores / num_scores
            #print(f"La media dei voti 'Overall' umani è: {average_scores}") 
        else:
            average_scores = None

        #Salvataggio output
        output.append({
            "context": elem["context"],
            "response": elem["response"],
            "overall_annotations": overall_annotations,
            "average_annotations": average_scores,
            "chateval_evaluation": evaluation
        })
        # TODO: time.sleep(15) per one-to-one
        # TODO: time.sleep(50) per simultaneous
        time.sleep(50)

    # Salvataggio in file json
    os.makedirs(args_output_dir, exist_ok=True)
    with open(os.path.join(args_output_dir, "fed_evaluation_results.json"), "w") as f:
        print(f"--- Valutazione completata. Salvo i risultati in {args_output_dir}/fed_evaluation_results.json ---")
        json.dump(output, f, indent=4)


elif "topical" in args_data_path:
    print(f"Topical")
    output = []
    n = 1
    for index, elem in enumerate(data[:n]):
        print(f"================================instance {index+1}====================================")
