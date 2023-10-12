import pandas as pd
from scipy import stats as stats


def calculate_memory_stats(hit, miss, cr, fa):
    pHit = hit / (hit + miss)
    # p(hit) = 0.0 or p(hit) = 1.0 will break SDT calculations. Fix this
    pHit_ = min(pHit, 1 - 1 / (2 * (hit + miss)))
    pHit_ = max(pHit_, 1 / (2 * (hit + miss)))
    zHit = stats.norm.ppf(pHit_)

    pCR = cr / (cr + fa)
    pFA = 1 - pCR
    # Same fix for the proportion of correct rejections
    pCR_ = min(pCR, 1 - 1 / (2 * (cr + fa)))
    pCR_ = max(pCR_, 1 / (2 * (cr + fa)))
    zCR = stats.norm.ppf(pCR_)
    zFA = -zCR

    d_prime = zHit - zFA # Sensitivity score, see SDT research
    c = - 1/2 * (zHit + zFA) # Criterion score, see SDT research
    corr_acc = pHit - pFA # Corrected accuracy (a.k.a., corrected recognition)

    return pHit, pFA, corr_acc, zHit, zFA, d_prime, c


def process_retrieval_responses(df, val_cue, val_tar, answer):
    df = df[(df['valence_cue'] == val_cue) & (df['valence_tar'] == val_tar)]
    df_signal = df[df['BA_answer'] == answer]
    hit = df_signal[df_signal['BA_resp'] == answer].shape[0]
    miss = df_signal[(df_signal['BA_resp'] != answer)].shape[0]
    df_noise = df[df['BA_answer'] != answer]
    fa = df_noise[df_noise['BA_resp'] == answer].shape[0] # false alarm
    cr = len(df_noise) - fa # correct_rejection
    return hit, miss, cr, fa


def get_subj_scores(df):
    memory_scores = {}
    for e_cue in ['emo', 'neu']:
        for e_tar in ['emo', 'neu']:
            if e_cue == e_tar:
                continue
            for answer in ['A', 'B']:
                # Frequencies. CR = correct rejection, FA = false alarm
                hit, miss, cr, fa = \
                    process_retrieval_responses(df, e_cue, e_tar, answer)

                # Proportions and SDT scores
                pHit, pFA, corr_acc, zHit, zFA, d_prime, c = \
                    calculate_memory_stats(hit, miss, cr, fa)
                memory_scores[f'{e_cue}_{e_tar}_{answer}_phit'] = pHit
                memory_scores[f'{e_cue}_{e_tar}_{answer}_pFA'] = pFA
                memory_scores[f'{e_cue}_{e_tar}_{answer}_corr_acc'] = corr_acc
                memory_scores[f'{e_cue}_{e_tar}_{answer}_zhit'] = zHit
                memory_scores[f'{e_cue}_{e_tar}_{answer}_zFA'] = zFA
                memory_scores[f'{e_cue}_{e_tar}_{answer}_d'] = d_prime
                memory_scores[f'{e_cue}_{e_tar}_{answer}_c'] = c
                memory_scores[f'{e_cue}_{e_tar}_{answer}_cd'] = c * d_prime
    memory_scores['id'] = df['id'].iloc[0]
    return memory_scores


def get_df_scores():
    fp_in = r'EmoTemp_replication_data.csv'
    df_trials = pd.read_csv(fp_in)
    df_trials = df_trials[~df_trials['excluded']]

    l_results = []
    for sn, df_subj in df_trials.groupby('id'):
        l_results.append(get_subj_scores(df_subj))
    df_scores = pd.DataFrame(l_results)
    return df_scores