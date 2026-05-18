#!/usr/bin/env python3
"""PCR² Main Experiment Runner — v2 (all audit fixes applied)
Run: python src/run_experiment.py (from repo root)"""
import sys, os, json, re
import numpy as np
from pathlib import Path
from scipy.special import expit
from scipy.stats import wilcoxon, spearmanr, bootstrap
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

np.random.seed(42)
ROOT = Path(__file__).resolve().parents[1]

K = 8; N_USERS = 100; SPARSITY = 0.35; N_PCS = 3

with open(ROOT / 'data' / 'wiki_corpus.json') as f:
    articles = json.load(f)
LEVEL_TO_B = {'easy': -1.5, 'medium': 0.0, 'hard': 1.5, 'expert': 3.0}

def compute_lltm_features(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) < 2: sentences = [text[:len(text)//2], text[len(text)//2:]]
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 5: return np.zeros(5)
    sl = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    return np.array([np.std(sl)/(np.mean(sl)+1e-6), sum(1 for w in words if len(w)>8)/len(words),
                     np.mean(sl)/30.0, 1.0-(len(set(words))/len(words)), np.mean([len(w) for w in words])/10.0])

def F(theta, b): p = expit(theta - b); return p * (1 - p)
def topk(a, k): return np.argsort(-a)[:k]

print("[1/7] Loading corpus...")
Q_raw = np.array([compute_lltm_features(a['text']) for a in articles])
b_assigned = np.array([LEVEL_TO_B[a['level']] for a in articles])
n_passages = len(articles)
lltm_w = np.array([1.5, 0.8, 0.6, 1.2, 0.3])
b_lltm_raw = Q_raw @ lltm_w
cal = LinearRegression().fit(b_lltm_raw.reshape(-1, 1), b_assigned)
b_lltm = cal.predict(b_lltm_raw.reshape(-1, 1))
rho_lltm, _ = spearmanr(b_lltm, b_assigned)
print(f"   {n_passages} articles, LLTM rho={rho_lltm:.3f}")

print("[2/7] Generating users (5 covariates, NO prior_correct)...")
true_thetas = np.concatenate([
    np.random.normal(-2.0,0.3,15), np.random.normal(-1.0,0.3,20),
    np.random.normal(0.0,0.4,25), np.random.normal(1.0,0.3,20),
    np.random.normal(2.0,0.4,12), np.random.normal(3.0,0.3,8),
])[:N_USERS]

Z_raw = np.column_stack([
    np.clip(12 + 2*true_thetas + np.random.normal(0,1.5,N_USERS), 6, 22),
    np.clip(3 + true_thetas + np.random.normal(0,1.2,N_USERS), 0, 7),
    np.clip(0.3 + 0.15*true_thetas + np.random.normal(0,0.12,N_USERS), 0, 1),
    np.clip(25 + 5*true_thetas + np.random.normal(0,8,N_USERS), 12, 70),
    np.clip(50 + 10*true_thetas + np.random.normal(0,10,N_USERS), 20, 100),
])
scaler = StandardScaler(); Z_scaled = scaler.fit_transform(Z_raw)
pca = PCA(n_components=N_PCS); Z_pc = pca.fit_transform(Z_scaled)

print("[3/7] Fitting PCR-IRT...")
response_matrix = np.zeros((N_USERS, n_passages))
mask = np.ones((N_USERS, n_passages), dtype=bool)
for i in range(N_USERS):
    for j in range(n_passages):
        response_matrix[i,j] = np.random.binomial(1, expit(true_thetas[i]-b_assigned[j]))
    hide = np.random.choice(n_passages, int(SPARSITY*n_passages), replace=False)
    mask[i, hide] = False

def pcr_irt_nll(params, R, msk, Z, n_p, n_i, n_c):
    g=params[:n_c]; e=params[n_c:n_c+n_p]; b=params[n_c+n_p:]
    th=Z@g+e; lo=th[:,None]-b[None,:]; p=expit(lo); p=np.clip(p,1e-8,1-1e-8)
    Rc=np.where(msk,R,0)
    ll=np.sum(msk*(Rc*np.log(p)+(1-Rc)*np.log(1-p)))
    ll-=0.02*np.sum(g**2)+0.03*np.sum(e**2)+0.01*np.sum(b**2)
    return -ll

res=minimize(pcr_irt_nll,np.zeros(N_PCS+N_USERS+n_passages),
             args=(response_matrix,mask,Z_pc,N_USERS,n_passages,N_PCS),
             method='L-BFGS-B',options={'maxiter':500})
gamma_hat=res.x[:N_PCS]; eps_hat=res.x[N_PCS:N_PCS+N_USERS]
theta_cold=Z_pc@gamma_hat
rho_cold=np.corrcoef(theta_cold,true_thetas)[0,1]
r2_person=np.var(theta_cold)/(np.var(theta_cold)+np.var(eps_hat)+1e-10)
print(f"   rho(cold)={rho_cold:.3f}, R2={r2_person:.3f}")

print("[4/7] Running all strategies...")
strats = {}
for i in range(N_USERS):
    F_o=F(true_thetas[i],b_assigned); F_c=F(theta_cold[i],b_assigned)
    F_l=F(theta_cold[i],b_lltm)
    # Simulated relevance oracle (NOT real BM25 — Gaussian centered on true θ)
    rel=np.exp(-0.5*((b_assigned-true_thetas[i])/2.0)**2)+np.random.normal(0,0.05,n_passages)
    rel=np.clip(rel,0,1)
    sel_o=topk(F_o,K); sel_c=topk(F_c,K); sel_l=topk(F_l,K)
    sel_d=topk(b_assigned,K); sel_e=topk(-b_assigned,K)
    sel_r=np.random.choice(n_passages,K,replace=False); sel_b=topk(rel,K)
    rel_pool=topk(rel,K*2); sel_h=rel_pool[topk(F(theta_cold[i],b_assigned[rel_pool]),K)]
    for nm,sel in [('Oracle',sel_o),('PCR_cold',sel_c),('PCR_LLTM',sel_l),
                   ('SimRel_oracle',sel_b),('Hybrid',sel_h),('Easiest_only',sel_e),
                   ('Diff_only',sel_d),('Random',sel_r)]:
        strats.setdefault(nm,[]).append(float(np.sum(F_o[sel])))

print("[5/7] Computing statistics and bootstrap CIs...")
oracle_mean=np.mean(strats['Oracle'])
results = {}
rng = np.random.default_rng(42)
for nm in strats:
    vals=np.array(strats[nm])
    r = {'mean':float(np.mean(vals)),'std':float(np.std(vals)),
         'pct_oracle':float(np.mean(vals)/oracle_mean*100)}
    # Bootstrap 95% CI
    boot = bootstrap((vals,), np.mean, n_resamples=10000, random_state=rng, confidence_level=0.95)
    r['ci_lo'] = float(boot.confidence_interval.low)
    r['ci_hi'] = float(boot.confidence_interval.high)
    # Cohen's d and p vs Difficulty-only (skip self-comparison)
    if nm != 'Diff_only':
        diff = vals - np.array(strats['Diff_only'])
        r['d_vs_diff'] = float(np.mean(diff) / (np.std(diff) + 1e-10))
        r['wins'] = int(np.sum(diff > 0))
        try:
            _, p = wilcoxon(vals, np.array(strats['Diff_only']), alternative='greater')
            r['p_vs_diff'] = float(p)
        except Exception:
            r['p_vs_diff'] = None  # strict JSON: null not NaN
    results[nm] = r

print("[6/7] Scaling and noise ablation...")
scaling = []
for sm in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    b_s=b_assigned*sm; fr_p=[]; fr_d=[]; fr_r=[]
    for i in range(N_USERS):
        Fo=F(true_thetas[i],b_s); Fc=F(theta_cold[i],b_s)
        fr_p.append(np.sum(Fo[topk(Fc,K)])); fr_d.append(np.sum(Fo[topk(b_s,K)]))
        fr_r.append(np.sum(Fo[np.random.choice(n_passages,K,replace=False)]))
    scaling.append({'sigma_b':float(np.std(b_s)),
                    'pcr_rand':float(np.mean(fr_p)/(np.mean(fr_r)+1e-10)),
                    'pcr_diff':float(np.mean(fr_p)/(np.mean(fr_d)+1e-10))})

ablation = []
for ns in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    theta_n=true_thetas+np.random.normal(0,ns,N_USERS)
    fo_l=[]; fn_l=[]; fd_l=[]
    for i in range(N_USERS):
        Fo=F(true_thetas[i],b_assigned); Fn=F(theta_n[i],b_assigned)
        fo_l.append(np.sum(Fo[topk(Fo,K)])); fn_l.append(np.sum(Fo[topk(Fn,K)]))
        fd_l.append(np.sum(Fo[topk(b_assigned,K)]))
    pct=np.mean(fn_l)/np.mean(fo_l)*100
    d_arr=np.array(fn_l)-np.array(fd_l); d=np.mean(d_arr)/(np.std(d_arr)+1e-10)
    ablation.append({'noise_sigma':ns,'pct_oracle':float(pct),'d_vs_diff':float(d)})

print("[7/7] Saving results...")
(ROOT / 'results').mkdir(exist_ok=True)
with open(ROOT/'results'/'main_results.json','w') as f:
    json.dump(results,f,indent=2)
with open(ROOT/'results'/'theta_estimation.json','w') as f:
    json.dump({'rho_cold':float(rho_cold),'r2_person':float(r2_person),
               'rho_lltm':float(rho_lltm),'n_users':N_USERS,'n_passages':n_passages,
               'pca_variance':pca.explained_variance_ratio_.tolist(),
               'gamma':gamma_hat.tolist(),'n_covariates':5,
               'covariates':['edu_years','read_freq','domain_fam','age','vocab_level']},f,indent=2)
with open(ROOT/'results'/'scaling.json','w') as f:
    json.dump(scaling,f,indent=2)
with open(ROOT/'results'/'ablation.json','w') as f:
    json.dump(ablation,f,indent=2)

print(f"\n{'='*60}\nRESULTS SUMMARY\n{'='*60}")
for nm in ['Oracle','PCR_cold','SimRel_oracle','Hybrid','PCR_LLTM','Easiest_only','Diff_only','Random']:
    r=results[nm]
    d_str=f"d={r['d_vs_diff']:.2f}" if 'd_vs_diff' in r else "     "
    ci_str=f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
    print(f"  {nm:<16s}  {r['mean']:.3f}±{r['std']:.3f}  {r['pct_oracle']:5.1f}%  {d_str}  {ci_str}")
print(f"\nTheta: rho={rho_cold:.3f}, R2={r2_person:.3f}")
print(f"LLTM: rho={rho_lltm:.3f}")
print(f"Covariates: edu_years, read_freq, domain_fam, age, vocab_level (NO prior_correct)")
print("Done.")
