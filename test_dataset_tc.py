import os
import json
import pandas as pd
import numpy as np

def get_data(file_path):
    """
    Carica un file JSON da un percorso specificato.
    Restituisce i dati decodificati o None se si verifica un errore.
    """
    if not os.path.exists(file_path):
        print(f"Errore: Il file '{file_path}' non è stato trovato.")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except json.JSONDecodeError:
        print(f"Errore: Impossibile decodificare il file JSON '{file_path}'. Controlla se è formattato correttamente.")
        return None

def validate_data_completeness(data):
    """
    Analizza i dati caricati e conta le istanze con campi mancanti o malformati.
    
    Restituisce (in ordine):
        (int): Conteggio delle istanze senza 'fact'.
        (int): Conteggio delle istanze senza 'responses' (chiave mancante o lista vuota).
        (int): Conteggio delle istanze con almeno una stringa di risposta vuota ("").
        (int): Conteggio delle istanze in cui len(responses) != 6.
    """
    if not isinstance(data, list):
        print("Errore: I dati di input non sono una lista di oggetti.")
        return 0, 0, 0, 0
        
    no_fact_count = 0
    no_responses_count = 0
    empty_response_string_count = 0
    wrong_responses_length_count = 0
    
    for item in data:
        # 1. Controlla 'fact' (None o stringa vuota)
        if not item.get("fact"):
            no_fact_count += 1
            
        # 2. Controlla 'responses' (None o lista vuota [])
        responses_list = item.get("responses")
        if not responses_list:
            no_responses_count += 1
        else:
            # Se la lista 'responses' esiste, controlla la lunghezza e il contenuto
            
            # 3. Controlla se la lunghezza è diversa da 6
            if len(responses_list) != 6:
                wrong_responses_length_count += 1
            
            # 4. Controlla se c'è almeno una risposta vuota all'interno della lista
            has_empty_response = False
            for response_item in responses_list:
                # Controlla se la chiave "response" è mancante, None, o una stringa vuota ""
                if not response_item.get("response"):
                    has_empty_response = True
                    break  # Trovata una, basta per contare l'istanza e passare alla prossima
            
            if has_empty_response:
                empty_response_string_count += 1
                
    return no_fact_count, no_responses_count, empty_response_string_count, wrong_responses_length_count


if __name__ == "__main__":
    input_path = "./agentverse/tasks/llm_eval/data/topical/tc_usr_data.json"
    
    data = get_data(input_path)
    
    if data:
        print(f"Input: {len(data)} istanze totali caricate.")
        missing_facts, missing_responses_list, empty_responses, wrong_length = validate_data_completeness(data) 
        print("\n--- Validazione Dati Topical-Chat ---")
        print(f"Istanze senza 'fact' validi: {missing_facts}")
        print(f"Istanze senza 'responses' (o lista vuota): {missing_responses_list}")
        print(f"Istanze con almeno una stringa 'response' vuota: {empty_responses}")
        print(f"Istanze con numero di 'responses' diverso da 6: {wrong_length}")