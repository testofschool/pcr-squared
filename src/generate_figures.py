#!/usr/bin/env python3
"""Generate all figures for the PCR² paper. Run from repo root: python src/generate_figures.py"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams["pdf.fonttype"] = 42  # Type 1 fonts, no Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from pathlib import Path
import json, re

np.random.seed(42)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.figsize': (5.5, 3.5), 'font.family': 'serif'})

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / 'figures'
FIGDIR.mkdir(exist_ok=True)
DATA_PATH = ROOT / 'data' / 'wiki_corpus.json'

# ========== Load data and reproduce experiment ==========
with open(DATA_PATH) as f:
    articles = json.load(f)

level_to_b = {'easy': -1.5, 'medium': 0.0, 'hard': 1.5, 'expert': 3.0}

def compute_lltm_features(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) < 2:
        sentences = [text[:len(text)//2], text[len(text)//2:]]
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 5:
        return np.zeros(5)
    sent_lens = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    return np.array([
        np.std(sent_lens) / (np.mean(sent_lens) + 1e-6),
        sum(1 for w in words if len(w) > 8) / len(words),
        np.mean(sent_lens) / 30.0,
        1.0 - (len(set(words)) / len(words)),
        np.mean([len(w) for w in words]) / 10.0,
    ])

Q_raw = np.array([compute_lltm_features(a['text']) for a in articles])
b_assigned = np.array([level_to_b[a['level']] for a in articles])
n_passages = len(articles)
lltm_w = np.array([1.5, 0.8, 0.6, 1.2, 0.3])
b_lltm_raw = Q_raw @ lltm_w
cal = LinearRegression().fit(b_lltm_raw.reshape(-1, 1), b_assigned)
b_lltm = cal.predict(b_lltm_raw.reshape(-1, 1))

n_users = 100
true_thetas = np.concatenate([
    np.random.normal(-2.0, 0.3, 15), np.random.normal(-1.0, 0.3, 20),
    np.random.normal(0.0, 0.4, 25), np.random.normal(1.0, 0.3, 20),
    np.random.normal(2.0, 0.4, 12), np.random.normal(3.0, 0.3, 8),
])[:n_users]

# 5 covariates (NO prior_correct)
Z_raw = np.column_stack([
    np.clip(12 + 2*true_thetas + np.random.normal(0,1.5,n_users), 6, 22),
    np.clip(3 + true_thetas + np.random.normal(0,1.2,n_users), 0, 7),
    np.clip(0.3 + 0.15*true_thetas + np.random.normal(0,0.12,n_users), 0, 1),
    np.clip(25 + 5*true_thetas + np.random.normal(0,8,n_users), 12, 70),
    np.clip(50 + 10*true_thetas + np.random.normal(0,10,n_users), 20, 100),
])
scaler = StandardScaler(); Z_s = scaler.fit_transform(Z_raw)
pca = PCA(n_components=3); Z_pc = pca.fit_transform(Z_s)

response_matrix = np.zeros((n_users, n_passages))
mask = np.ones((n_users, n_passages), dtype=bool)
for i in range(n_users):
    for j in range(n_passages):
        response_matrix[i,j] = np.random.binomial(1, expit(true_thetas[i]-b_assigned[j]))
    hide = np.random.choice(n_passages, int(0.35*n_passages), replace=False)
    mask[i, hide] = False

def nll(params, R, msk, Z, np_, ni, nc):
    g=params[:nc]; e=params[nc:nc+np_]; b=params[nc+np_:]
    th=Z@g+e; lo=th[:,None]-b[None,:]; p=expit(lo); p=np.clip(p,1e-8,1-1e-8)
    Rc=np.where(msk,R,0)
    ll=np.sum(msk*(Rc*np.log(p)+(1-Rc)*np.log(1-p)))
    ll-=0.02*np.sum(g**2)+0.03*np.sum(e**2)+0.01*np.sum(b**2)
    return -ll

res=minimize(nll,np.zeros(3+n_users+n_passages),args=(response_matrix,mask,Z_pc,n_users,n_passages,3),method='L-BFGS-B',options={'maxiter':500})
gamma_hat=res.x[:3]; eps_hat=res.x[3:3+n_users]
theta_cold=Z_pc@gamma_hat

def F(theta,b,a=1.0): p=expit(a*(theta-b)); return p*(1-p)
def topk(arr,k): return np.argsort(-arr)[:k]
K=8

strats = {}
for i in range(n_users):
    F_o=F(true_thetas[i],b_assigned); F_c=F(theta_cold[i],b_assigned)
    sel_o=topk(F_o,K); sel_c=topk(F_c,K)
    sel_d=topk(b_assigned,K); sel_r=np.random.choice(n_passages,K,replace=False)
    for nm,sel in [('Oracle',sel_o),('PCR cold',sel_c),('Diff-only',sel_d),('Random',sel_r)]:
        strats.setdefault(nm,[]).append(np.sum(F_o[sel]))

# ========== FIGURE 1 ==========
fig, ax = plt.subplots(figsize=(5.5, 3.2))
b_range = np.linspace(-4, 4, 300)
for theta, ls, label in [(-2,'-.','$\\theta=-2$ (elementary)'),(0,'--','$\\theta=0$ (high school)'),(2,'-','$\\theta=+2$ (undergraduate)')]:
    ax.plot(b_range, F(theta, b_range), ls, linewidth=1.8, label=label)
ax.set_xlabel('Content Difficulty ($b$)'); ax.set_ylabel('$\\mathcal{F}(\\theta, b) = P(1-P)$')
ax.set_title('Frontier Value Function'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(FIGDIR/'fig1_frontier_value.pdf'); plt.close(); print("  fig1")

# ========== FIGURE 2 ==========
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
axes[0].scatter(true_thetas, theta_cold, s=12, alpha=0.6, c='#2166ac')
axes[0].plot([-3,4],[-3,4],'k--',alpha=0.3)
axes[0].set_xlabel('True $\\theta$'); axes[0].set_ylabel('Estimated $\\hat\\theta$ (cold-start)')
rho=np.corrcoef(theta_cold,true_thetas)[0,1]; axes[0].set_title(f'$\\rho = {rho:.3f}$'); axes[0].grid(True,alpha=0.3)
axes[1].bar(range(1,4),pca.explained_variance_ratio_[:3],color='#2166ac',alpha=0.7)
axes[1].set_xlabel('Principal Component'); axes[1].set_ylabel('Variance Explained')
axes[1].set_title(f'PCA: {pca.explained_variance_ratio_[:3].sum()*100:.1f}% in 3 PCs')
axes[1].set_xticks([1,2,3]); axes[1].grid(True,alpha=0.3,axis='y')
plt.tight_layout(); plt.savefig(FIGDIR/'fig2_theta_estimation.pdf'); plt.close(); print("  fig2")

# ========== FIGURE 3 (shortened title) ==========
fig, ax = plt.subplots(figsize=(5.5, 3.2))
colors = {'easy':'#4daf4a','medium':'#377eb8','hard':'#ff7f00','expert':'#e41a1c'}
for a in articles:
    ax.scatter(level_to_b[a['level']], compute_lltm_features(a['text'])@lltm_w, c=colors[a['level']], s=25, alpha=0.7, edgecolors='none')
for lvl,c in colors.items(): ax.scatter([],[],c=c,s=25,label=lvl)
rho_l,_=spearmanr(b_lltm,b_assigned)
ax.set_xlabel('Assigned Difficulty ($b$)'); ax.set_ylabel('LLTM Score ($\\hat{b}_{\\mathrm{LLTM}}$)')
ax.set_title(f'LLTM on Wikipedia ($\\rho = {rho_l:.3f}$)')
ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig(FIGDIR/'fig3_lltm_failure.pdf'); plt.close(); print("  fig3")

# ========== FIGURE 4 (load from canonical JSON) ==========
fig, ax = plt.subplots(figsize=(5.5, 3.2))
json_path = ROOT / 'results' / 'main_results.json'
if json_path.exists():
    with open(json_path) as jf:
        jr = json.load(jf)
    names_j = ['Oracle', 'PCR_cold', 'Diff_only', 'Random']
    labels_j = ['Oracle PCR\n($\\theta$ known)', 'PCR cold-start\n($\\hat\\theta = Z\\gamma$)', 'Difficulty-only', 'Random']
    means_j = [jr[n]['mean'] for n in names_j]
    stds_j = [jr[n]['std'] for n in names_j]
else:
    names_j = ['Oracle','PCR cold','Diff-only','Random']
    means_j = [np.mean(strats[n]) for n in names_j]; stds_j = [np.std(strats[n]) for n in names_j]
    labels_j = ['Oracle PCR\n($\\theta$ known)','PCR cold-start\n($\\hat\\theta = Z\\gamma$)','Difficulty-only','Random']
clrs=['#7570b3','#1b9e77','#d95f02','#999999']
ax.bar(range(len(names_j)),means_j,yerr=stds_j,color=clrs,capsize=4,alpha=0.85)
ax.set_xticks(range(len(names_j)))
ax.set_xticklabels(labels_j,fontsize=9)
ax.set_ylabel('Frontier Value Captured'); ax.set_title(f'{K} of {n_passages} Passages, {n_users} Users')
ax.grid(True,alpha=0.3,axis='y')
for i,(m,s) in enumerate(zip(means_j,stds_j)):
    ax.text(i,m+s+0.03,f'{m/means_j[0]*100:.0f}%',ha='center',fontsize=8)
plt.tight_layout(); plt.savefig(FIGDIR/'fig4_strategy_comparison.pdf'); plt.close(); print("  fig4")

# ========== FIGURE 5 ==========
fig, ax = plt.subplots(figsize=(5.5, 3.5))
for yi,theta_s in enumerate([-2.0,-1.0,0.0,1.0,2.0,3.0]):
    F_s=F(theta_s,b_assigned); sel=topk(F_s,K)
    ax.scatter(b_assigned[sel],[yi]*K,c='#1b9e77',s=40,zorder=3,alpha=0.8)
    ax.scatter([theta_s],[yi],marker='x',c='#d95f02',s=80,zorder=4,linewidths=2)
ax.set_yticks(range(6)); ax.set_yticklabels([f'$\\theta={t:+.0f}$' for t in [-2,-1,0,1,2,3]])
ax.set_xlabel('Selected Passage Difficulty ($b$)')
ax.set_title(f'PCR Selections per Ability Level (top-{K} of {n_passages})')
ax.legend(['Selected passages','User $\\theta$'],loc='lower right',fontsize=8,markerscale=0.8)
ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig(FIGDIR/'fig5_per_ability.pdf'); plt.close(); print("  fig5")

# ========== FIGURE 6 (LOG SCALE — from canonical JSON) ==========
fig, ax = plt.subplots(figsize=(5.5, 3.2))
scaling_path = ROOT / 'results' / 'scaling.json'
if scaling_path.exists():
    with open(scaling_path) as jf:
        sc = json.load(jf)
    sigma_vals = [s['sigma_b'] for s in sc]
    ratio_rand = [s['pcr_rand'] for s in sc]
    ratio_diff = [s['pcr_diff'] for s in sc]
else:
    sigmas=[0.25,0.5,1.0,1.5,2.0,3.0]; ratio_rand=[]; ratio_diff=[]; sigma_vals=[]
    for sm in sigmas:
        b_s=b_assigned*sm; fr_p=[]; fr_d=[]; fr_r=[]
        for i in range(n_users):
            Fo=F(true_thetas[i],b_s); Fc=F(theta_cold[i],b_s)
            fr_p.append(np.sum(Fo[topk(Fc,K)])); fr_d.append(np.sum(Fo[topk(b_s,K)]))
            fr_r.append(np.sum(Fo[np.random.choice(n_passages,K,replace=False)]))
        sigma_vals.append(np.std(b_s))
        ratio_rand.append(np.mean(fr_p)/np.mean(fr_r))
        ratio_diff.append(np.mean(fr_p)/(np.mean(fr_d)+1e-10))
ax.plot(sigma_vals,ratio_rand,'o-',c='#1b9e77',label='PCR / Random',linewidth=1.5)
ax.plot(sigma_vals,ratio_diff,'s-',c='#d95f02',label='PCR / Difficulty-only',linewidth=1.5)
ax.axhline(1.0,color='gray',linestyle='--',alpha=0.5)
ax.set_xlabel('Content Difficulty Diversity ($\\sigma_b$)'); ax.set_ylabel('PCR Advantage Ratio')
ax.set_title('PCR Advantage Scales with Difficulty Diversity')
ax.set_yscale('log'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig(FIGDIR/'fig6_scaling.pdf'); plt.close(); print("  fig6")

# ========== FIGURE 7 (from canonical JSON) ==========
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ablation_path = ROOT / 'results' / 'ablation.json'
if ablation_path.exists():
    with open(ablation_path) as jf:
        ab = json.load(jf)
    noise_sigmas = [a['noise_sigma'] for a in ab]
    pct_oracle = [a['pct_oracle'] for a in ab]
    d_vs_diff = [a['d_vs_diff'] for a in ab]
else:
    noise_sigmas=[0.0,0.5,1.0,1.5,2.0,3.0]; pct_oracle=[]; d_vs_diff=[]
    for ns in noise_sigmas:
        theta_n=true_thetas+np.random.normal(0,ns,n_users)
        fo_l=[]; fn_l=[]; fd_l=[]
        for i in range(n_users):
            Fo=F(true_thetas[i],b_assigned); Fn=F(theta_n[i],b_assigned)
            fo_l.append(np.sum(Fo[topk(Fo,K)])); fn_l.append(np.sum(Fo[topk(Fn,K)]))
            fd_l.append(np.sum(Fo[topk(b_assigned,K)]))
        pct_oracle.append(np.mean(fn_l)/np.mean(fo_l)*100)
        d_arr=np.array(fn_l)-np.array(fd_l); d_vs_diff.append(np.mean(d_arr)/(np.std(d_arr)+1e-10))
ax2=ax.twinx()
l1,=ax.plot(noise_sigmas,pct_oracle,'o-',c='#1b9e77',linewidth=1.5,label='% Oracle')
l2,=ax2.plot(noise_sigmas,d_vs_diff,'s--',c='#d95f02',linewidth=1.5,label="Cohen's $d$ vs Diff-only")
ax.set_xlabel('$\\theta$ Estimation Noise ($\\sigma$)')
ax.set_ylabel('% of Oracle Quality',color='#1b9e77')
ax2.set_ylabel("Cohen's $d$ vs Difficulty-only",color='#d95f02')
ax.set_title('Robustness to $\\theta$ Estimation Noise')
ax.legend(handles=[l1,l2],fontsize=9,loc='center right'); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig(FIGDIR/'fig7_robustness.pdf'); plt.close(); print("  fig7")

print("All figures generated (Type 1 fonts, relative paths, log-scale Fig 6).")
