import numpy as np
import json
import os
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import cohen_kappa_score

def get_human_scores(path):
    scores_list = []
    
    if not os.path.exists(path):
        print(f"Errore: Il file non è stato trovato al percorso: {path}")
        return np.array([])
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        score = item.get("average_annotations")
        if score is not None:
            scores_list.append(score)

    return np.array(scores_list)

def get_agent_scores(path):
    scores_list = []
    
    if not os.path.exists(path):
        print(f"Errore: Il file non è stato trovato al percorso: {path}")
        return np.array([])
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        evs = item.get("chateval_evaluation")
        score_sum = 0
        voti_validi = 0 # Contatore per gestire correttamente i 'null'
        
        for e in evs:
            score_agent = e.get("score") 
            
            # Tenta di estrarre il punteggio dal testo se 'score' è 'null'
            if score_agent is None:
                evaluation_text = e.get("evaluation", "")
                # Cerca 'Overall Score: [numero]' (anche con decimali)
                match = re.search(r'Overall Score:\s*(\d+\.?\d*)', evaluation_text) 
                if match:
                    try:
                        score_agent = float(match.group(1))
                    except ValueError:
                        score_agent = None # Fallito il cast
            
            if isinstance(score_agent, (int, float)):
                score_sum += score_agent
                voti_validi += 1
                
        # Calcola la media solo sui voti validi
        score_avg = score_sum / voti_validi if voti_validi > 0 else 0
        scores_list.append(score_avg)

    return np.array(scores_list)

# --- NUOVA FUNZIONE AGGIUNTA ---
def calculate_kappa(human_scores, agent_scores):
    """
    Arrotonda i punteggi continui a interi e calcola il Kappa di Cohen.
    """
    # 1. Arrotonda i punteggi float all'intero più vicino (es. 2.6 -> 3)
    human_rounded = np.round(human_scores).astype(int)
    agent_rounded = np.round(agent_scores).astype(int)
    
    # 2. Calcola Kappa sui dati arrotondati (ora categoriali)
    # 'weights="quadratic"' è spesso usato per scale ordinali (0-5) 
    # per penalizzare di più gli errori grandi (es. 1 vs 5) rispetto a quelli piccoli (es. 3 vs 4).
    # Usa 'weights=None' (default) per il Kappa standard non pesato.
    kappa = cohen_kappa_score(human_rounded, agent_rounded, weights="quadratic")
    
    return kappa
# --- FINE NUOVA FUNZIONE ---


def main():
    # path_1 = "./outputs/fed/one-to-one/results.json"
    # path_2 = "./outputs/fed/simultaneous/results.json"
    # path_3 = "./outputs/fed/simultaneous_sum/results.json"
    path_1 = "./outputs/topical/one-to-one/results.json"
    path_2 = "./outputs/topical/simultaneous/results.json"
    path_3 = "./outputs/topical/simultaneous_sum/results.json"

    human_scores = get_human_scores(path_1) 
    
    one_to_one_scores = get_agent_scores(path_1)
    sim_scores = get_agent_scores(path_2)
    sim_sum_scores = get_agent_scores(path_3)

    # One to One correlations
    print("\nCorrelazioni One-to-One:")
    corr_spearman, _ = spearmanr(human_scores, one_to_one_scores)
    print(f"Spearman (rho) One-to-One: {corr_spearman:.3f}")
    corr_kendall, _ = kendalltau(human_scores, one_to_one_scores)
    print(f"Kendall-Tau (tau) One-to-One: {corr_kendall:.3f}")
    corr_pearson, _ = pearsonr(human_scores, one_to_one_scores)
    print(f"Pearson (r) One-to-One: {corr_pearson:.3f}")
    kappa_one_to_one = calculate_kappa(human_scores, one_to_one_scores)
    print(f"Cohen's Kappa (Quadratic) One-to-One: {kappa_one_to_one:.3f}") # <-- AGGIUNTO

    # Simultaneous correlations
    print("\nCorrelazioni Simultaneous:")
    corr_spearman, _ = spearmanr(human_scores, sim_scores)
    print(f"Spearman (rho) Simultaneous: {corr_spearman:.3f}")
    corr_kendall, _ = kendalltau(human_scores, sim_scores)
    print(f"Kendall-Tau (tau) Simultaneous: {corr_kendall:.3f}")
    corr_pearson, _ = pearsonr(human_scores, sim_scores)
    print(f"Pearson (r) Simultaneous: {corr_pearson:.3f}")
    kappa_sim = calculate_kappa(human_scores, sim_scores)
    print(f"Cohen's Kappa (Quadratic) Simultaneous: {kappa_sim:.3f}") # <-- AGGIUNTO

    # Simultaneous with summarizer correlations
    print("\nCorrelazioni Simultaneous with summarizer:")
    corr_spearman, _ = spearmanr(human_scores, sim_sum_scores)
    print(f"Spearman (rho) Simultaneous with summarizer: {corr_spearman:.3f}")
    corr_kendall, _ = kendalltau(human_scores, sim_sum_scores)
    print(f"Kendall-Tau (tau) Simultaneous with summarizer: {corr_kendall:.3f}")
    corr_pearson, _ = pearsonr(human_scores, sim_sum_scores)
    print(f"Pearson (r) Simultaneous with summarizer: {corr_pearson:.3f}")
    kappa_sim_sum = calculate_kappa(human_scores, sim_sum_scores)
    print(f"Cohen's Kappa (Quadratic) Simultaneous with summarizer: {kappa_sim_sum:.3f}") # <-- AGGIUNTO


if __name__ == "__main__":
    import re # Assicurati che 're' sia importato se usi la logica di fallback
    main()