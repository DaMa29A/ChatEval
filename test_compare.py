import json
import os

def load_json_file(file_path):
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


def compare_contexts(file_path1, file_path2):
    """
    Scorre due file JSON e confronta il campo "context" di ogni istanza.
    """
    print(f"--- Avvio confronto contesti tra:\n  1: {file_path1}\n  2: {file_path2}\n---")
    
    data1 = load_json_file(file_path1)
    data2 = load_json_file(file_path2)

    if data1 is None or data2 is None:
        print("Errore: Impossibile caricare uno o entrambi i file. Confronto annullato.")
        return

    # Controlla se i file hanno lo stesso numero di istanze
    if len(data1) != len(data2):
        print(f"ATTENZIONE: I file hanno un numero diverso di istanze ({len(data1)} vs {len(data2)}).")
        print("Il confronto verrà eseguito solo fino alla fine del file più corto.")

    discrepancies = 0
    
    # Usa zip() per scorrere entrambe le liste in parallelo
    # index parte da 0, quindi aggiungiamo 1 per la stampa
    for index, (item1, item2) in enumerate(zip(data1, data2)):
        
        context1 = item1.get("context")
        context2 = item2.get("context")

        if context1 is None or context2 is None:
            print(f"Istanza {index + 1}: ERRORE - Chiave 'context' mancante in uno dei due file.")
            discrepancies += 1
            continue

        # Confronta le stringhe del contesto
        if context1 == context2:
            print(f"Istanza {index + 1}: Contesti UGUALI.")
        else:
            print(f"Istanza {index + 1}: *** CONTESTI DIVERSI ***")
            discrepancies += 1
            
            # (Opzionale) Decommenta le righe seguenti se vuoi vedere i contesti diversi
            # print(f"  File 1: {context1[:50]}...")
            # print(f"  File 2: {context2[:50]}...")

    print("\n--- Confronto completato ---")
    if discrepancies == 0:
        print("Risultato: Tutti i contesti corrispondono perfettamente.")
    else:
        print(f"Risultato: Trovate {discrepancies} discrepanze nei contesti.")


# --- NUOVA FUNZIONE AGGIUNTA ---
def get_nth_context(data, n):
    """
    Recupera e stampa il campo 'context' dell'n-esimo elemento (basato su 1) 
    dalla lista di dati.
    """
    if not isinstance(data, list):
        print("Errore: I dati forniti non sono una lista.")
        return

    # Converti l'indice n (basato su 1) in indice 0 (basato su Python)
    index = n - 1

    # Controllo dei limiti
    if index < 0 or index >= len(data):
        print(f"Errore: L'indice {n} è fuori dai limiti. Il dataset ha {len(data)} elementi (da 1 a {len(data)}).")
        return
        
    # Recupera l'elemento
    item = data[index]
    
    # Recupera il contesto
    context = item.get("context")
    
    if context:
        print(f"\n--- Contesto per l'istanza {n} ---")
        print(context)
        print("---------------------------------")
        return context
    else:
        print(f"Errore: L'istanza {n} non ha un campo 'context' valido.")
        return None
# --- FINE NUOVA FUNZIONE ---

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



# --- NUOVA FUNZIONE AGGIUNTA ---
def search_by_context(data, search_string):
    """
    Cerca una stringa (search_string) all'interno del campo 'context' 
    di tutti gli elementi nei dati. Stampa i contesti che corrispondono.
    """
    if not isinstance(data, list):
        print("Errore: I dati forniti non sono una lista.")
        return

    found_count = 0
    
    context_ret = ""
    # Iteriamo su tutti gli elementi
    for index, item in enumerate(data):
        context = item.get("context")
        
        if context:
            # Controlliamo se la stringa cercata è contenuta nel contesto
            # .lower() rende la ricerca case-insensitive
            if search_string.lower() == context.lower():
                print(f"\n--- Trovata Corrispondenza (Istanza {index + 1}) ---")
                print(context)
                print("-----------------------------------")
                found_count += 1
        else:
            # Questo avviso aiuta a identificare dati corrotti
            print(f"Avviso: Istanza {index + 1} non ha un campo 'context' valido.")


    if found_count == 0:
        print(f"\nRicerca completata. Nessuna istanza trovata contenente: '{search_string}'")
    else:
        print(f"\nRicerca completata. Trovate {found_count} istanze corrispondenti.")
    


# --- NUOVA FUNZIONE AGGIUNTA ---
def save_first_n_elements(data, n, output_directory, original_filename):
    """
    Prende i primi 'n' elementi dai dati e li salva in un nuovo file JSON
    chiamato "results_n.json" (es. "results_10.json") nella directory specificata.
    """
    if not isinstance(data, list):
        print("Errore: I dati forniti non sono una lista.")
        return

    if n > len(data):
        print(f"Avviso: Richiesti {n} elementi, ma il dataset ne ha solo {len(data)}. Salvo tutti gli elementi.")
        n = len(data)

    # Prende solo i primi 'n' elementi
    sliced_data = data[:n]
    
    # Costruisce il nuovo nome del file
    base_name = os.path.basename(original_filename)
    new_filename = f"{os.path.splitext(base_name)[0]}_first_{n}.json"
    output_path = os.path.join(output_directory, new_filename)

    try:
        # Assicura che la directory di output esista
        os.makedirs(output_directory, exist_ok=True)
        
        # Salva i dati affettati nel nuovo file JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sliced_data, f, indent=4)
        print(f"\n--- Salvataggio completato ---")
        print(f"I primi {n} elementi sono stati salvati in: {output_path}")
        print("---------------------------------")
    except Exception as e:
        print(f"Errore durante il salvataggio del file JSON: {e}")
# --- FINE NUOVA FUNZIONE ---



# --- NUOVA FUNZIONE AGGIUNTA ---
def save_last_n_elements(data, n, output_directory, original_filename):
    """
    Prende gli ultimi 'n' elementi dai dati e li salva in un nuovo file JSON
    nella directory specificata.
    """
    if not isinstance(data, list):
        print("Errore: I dati forniti non sono una lista.")
        return

    if n > len(data):
        print(f"Avviso: Richiesti {n} elementi, ma il dataset ne ha solo {len(data)}. Salvo tutti gli elementi.")
        n = len(data)
    elif n <= 0:
        print("Errore: 'n' deve essere un numero positivo.")
        return

    # Usa lo slicing negativo per prendere gli ultimi 'n' elementi
    sliced_data = data[-n:]
    
    # Costruisce il nuovo nome del file
    base_name = os.path.basename(original_filename)
    new_filename = f"{os.path.splitext(base_name)[0]}_last_{n}.json"
    output_path = os.path.join(output_directory, new_filename)

    try:
        # Assicura che la directory di output esista
        os.makedirs(output_directory, exist_ok=True)
        
        # Salva i dati affettati nel nuovo file JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sliced_data, f, indent=4)
        print(f"\n--- Salvataggio 'Last N' completato ---")
        print(f"Gli ultimi {n} elementi sono stati salvati in: {output_path}")
        print("---------------------------------")
    except Exception as e:
        print(f"Errore during saving the JSON file: {e}")
# --- FINE NUOVA FUNZIONE ---


if __name__ == "__main__":
    # file di 385
    # i primi 100 coincidono    dalla 101 alla 110
    # gli ultimi 275 coincidono

    path_file1 = "outputs/fed/one-to-one/results.json"
    path_file2 = "outputs/fed/simultaneous/results.json"
    compare_contexts(path_file1, path_file2)