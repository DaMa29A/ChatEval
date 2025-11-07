import numpy as np
import json
import os
from scipy.stats import spearmanr, kendalltau, pearsonr

def get_human_scores(path):
    scores_list = []
    
    if not os.path.exists(path):
        print(f"Errore: Il file non è stato trovato al percorso: {path}")
        return np.array([])
     
    with open(path, 'r') as f:
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
    
    with open(path, 'r') as f:
        data = json.load(f)

    for item in data:
        evs = item.get("chateval_evaluation")
        score_sum = 0
        for e in evs:
            score_agent = e.get("score") 
            if isinstance(score_agent, (int, float)):
                score_sum += score_agent
        score_avg = score_sum / len(evs) if evs else 0
        scores_list.append(score_avg)

    return np.array(scores_list)


def main():
    path_1 = "outputs/fed/one-to-one/results.json"
    path_2 = "outputs/fed/simultaneous/results.json"
    path_3 = "outputs/fed/simultaneous_sum/fed_evaluation_results.json"

    human_scores = get_human_scores(path_1) 
    print(f"Human scores: {human_scores}")

    one_to_one_scores = get_agent_scores(path_1)
    print(f"Valutazione One-to-One: {one_to_one_scores}")

    sim_scores = get_agent_scores(path_2)
    print(f"Valutazione Simultaneous: {sim_scores}")

    # sim_sum_scores = get_agent_scores(path_3)
    # print(f"Valutazione Simultaneous with summarizer: {sim_sum_scores}")

    # One to One correlations
    print("\nCorrelazioni One-to-One:")
    corr_spearman, p_spearman = spearmanr(human_scores, one_to_one_scores)
    print(f"Spearman (rho) One-to-One: {corr_spearman:.3f}")
    corr_kendall, p_kendall = kendalltau(human_scores, one_to_one_scores)
    print(f"Kendall-Tau (tau) One-to-One: {corr_kendall:.3f}")
    corr_pearson, p_pearson = pearsonr(human_scores, one_to_one_scores)
    print(f"Pearson (r) One-to-One: {corr_pearson:.3f}")

    # Simultaneous correlations
    print("\nCorrelazioni Simultaneous:")
    corr_spearman, p_spearman = spearmanr(human_scores, sim_scores)
    print(f"Spearman (rho) Simultaneous: {corr_spearman:.3f}")
    corr_kendall, p_kendall = kendalltau(human_scores, sim_scores)
    print(f"Kendall-Tau (tau) Simultaneous: {corr_kendall:.3f}")
    corr_pearson, p_pearson = pearsonr(human_scores, sim_scores)
    print(f"Pearson (r) Simultaneous: {corr_pearson:.3f}")

    # # Simultaneous with summarizer correlations
    # print("\nCorrelazioni Simultaneous with summarizer:")
    # corr_spearman, p_spearman = spearmanr(human_scores, sim_sum_scores)
    # print(f"Spearman (rho) Simultaneous with summarizer: {corr_spearman:.3f}")
    # corr_kendall, p_kendall = kendalltau(human_scores, sim_sum_scores)
    # print(f"Kendall-Tau (tau) Simultaneous with summarizer: {corr_kendall:.3f}")
    # corr_pearson, p_pearson = pearsonr(human_scores, sim_sum_scores)
    # print(f"Pearson (r) Simultaneous with summarizer: {corr_pearson:.3f}")


if __name__ == "__main__":
    main()