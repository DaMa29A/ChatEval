import json
import os
from collections import defaultdict

def get_data(file_path):
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


def analyze_dataset(data):
    try:
        if isinstance(data, list):
            total_items = len(data)
            print(f"\n--- Analisi ---")
            print(f"Il dataset contiene {total_items} elementi.")

            contexts = set()
            systems = set()
            annotation_keys = set()
            no_ctx = 0
            no_sys = 0
            no_resp = 0
            data_with_response = []
            
            for item in data:
                if 'context' in item:
                    contexts.add(item['context'])
                else:
                    no_ctx +=1
                if 'system' in item:
                    systems.add(item['system'])
                else:
                    no_sys +=1
                if 'response' in item:
                    data_with_response.append(item)
                else:
                    no_resp +=1
                if 'annotations' in item and isinstance(item['annotations'], dict):
                    annotation_keys.update(item['annotations'].keys())

            print(f"Numero di contesti unici: {len(contexts)}")
            print(f"Numero di elem senza contesto: {no_ctx}")
            print(f"Numero di elem senza system: {no_sys}")
            print(f"Numero di elem senza response: {no_resp}")
            print(f"Sistemi trovati: \n{systems if systems else 'Nessuno'}")
            print(f"Chiavi di annotazione trovate: \n{annotation_keys if annotation_keys else 'Nessuna'}")

            print("--- Fine Analisi ---\n")
            return data_with_response
        else:
            print(f"Errore: Il dataset non contiene una lista JSON come struttura principale.")
            print(f"Tipo di dati trovato: {type(data)}")
            return None
    except Exception as e:
        print(f"Si è verificato un errore imprevisto durante l'analisi: {e}")
        return None


def save_data_to_json(data, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Dati salvati con successo in: {file_path}")
    
    except IOError as e:
        print(f"Errore: Impossibile scrivere nel file '{file_path}'. Errore IO: {e}")
    except TypeError as e:
        print(f"Errore: I dati non sono serializzabili in JSON. Errore: {e}")
    except Exception as e:
        print(f"Si è verificato un errore imprevisto durante il salvataggio: {e}")


if __name__ == "__main__":
    input_dataset_file = "./agentverse/tasks/llm_eval/data/fed/fed_data.json"
    original_data = get_data(input_dataset_file)

    data_with_response = analyze_dataset(original_data)
    print(f"Data with response: {len(data_with_response)}")
    data_with_response2 = analyze_dataset(data_with_response)

    output_dataset_file = "./agentverse/tasks/llm_eval/data/fed/fed_data_preproc.json"
    save_data_to_json(data_with_response, output_dataset_file)