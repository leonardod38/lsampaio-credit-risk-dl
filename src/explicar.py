"""
explicar.py — Fase 3: SHAP GradientExplainer (PyTorch).
Autor: Leonardo Sampaio
"""
import sys, mlflow, numpy as np, pandas as pd
import matplotlib.pyplot as plt, shap, torch
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=ROOT/"data"; MODELS_DIR=ROOT/"models"
FIGS_DIR=ROOT/"reports"/"figures"
TRACKING_URI="http://127.0.0.1:5000"
EXPERIMENT="classificacao-risco-credito"
CLASSES=["baixo","medio","alto","critico"]
FEATURES=["score_credito","renda_mensal","divida_total","ratio_divida_renda",
"historico_pagamento","tempo_emprego_anos","num_contas","num_atrasos",
"idade","score_normalizado","genero_enc","regiao_enc","faixa_etaria_enc"]

def carregar_modelo():
    sys.path.insert(0, str(ROOT/"src"))
    from train import CreditRiskNet
    pts=sorted(MODELS_DIR.glob("*.pt"))
    if not pts: print("Execute train.py primeiro."); sys.exit(1)
    m=CreditRiskNet(len(FEATURES)); m.load_state_dict(torch.load(str(pts[-1]),weights_only=True)); m.eval()
    print(f"Modelo: {pts[-1].name}"); return m

def pipeline_explicar():
    mlflow.set_tracking_uri(TRACKING_URI); mlflow.set_experiment(EXPERIMENT)
    modelo=carregar_modelo()
    df_tr=pd.read_csv(DATA_DIR/"credito_train.csv")
    df_te=pd.read_csv(DATA_DIR/"credito_test.csv")
    X_tr=torch.FloatTensor(df_tr[FEATURES].values)
    X_te=torch.FloatTensor(df_te[FEATURES].values[:500])
    X_np=X_te.numpy()
    y_te=df_te["label"].values[:500]
    bg=X_tr[np.random.choice(len(X_tr),200,replace=False)]
    print("Calculando SHAP values...")
    exp=shap.GradientExplainer(modelo,bg)
    sv=exp.shap_values(X_te)
    # sv pode ser lista ou array 3D
    def get_sv(i):
        return sv[i] if isinstance(sv,list) else sv[:,:,i]
    FIGS_DIR.mkdir(parents=True,exist_ok=True); arts=[]
    # 1 summary global
    sv_list=[get_sv(i) for i in range(len(CLASSES))]
    shap.summary_plot(sv_list,X_np,feature_names=FEATURES,class_names=CLASSES,show=False,max_display=13)
    p=str(FIGS_DIR/"shap_summary_global.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close(); arts.append(p)
    print("  Summary global OK")
    # 2 summary por classe
    for i,c in enumerate(CLASSES):
        shap.summary_plot(sv_list[i],X_np,feature_names=FEATURES,show=False,max_display=10)
        plt.title(f"SHAP — {c.upper()}",fontsize=12)
        p=str(FIGS_DIR/f"shap_summary_{c}.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close(); arts.append(p)
    print("  Summary por classe OK")
    # 3 waterfall por classe
    for i,c in enumerate(CLASSES):
        idxs=np.where(y_te==i)[0]
        if not len(idxs): continue
        idx=idxs[0]; vals=sv_list[i][idx]
        e=shap.Explanation(values=vals,base_values=float(sv_list[i].mean()),data=X_np[idx],feature_names=FEATURES)
        shap.plots.waterfall(e,show=False,max_display=10)
        plt.title(f"Waterfall — {c.upper()}",fontsize=11)
        p=str(FIGS_DIR/f"shap_waterfall_{c}.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close(); arts.append(p)
    print("  Waterfall OK")
    # 4 feature importance
    ms=np.mean([np.abs(s).mean(0) for s in sv_list],axis=0)
    imp=pd.Series(ms,index=FEATURES).sort_values(ascending=True)
    fig,ax=plt.subplots(figsize=(9,6))
    imp.plot(kind="barh",ax=ax,color=["#D85A30" if v>imp.median() else "#378ADD" for v in imp],edgecolor="none")
    ax.set_title("Feature Importance Global (SHAP)",fontsize=13); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); p=str(FIGS_DIR/"shap_feature_importance.png"); plt.savefig(p,dpi=150); plt.close(); arts.append(p)
    print("  Feature importance OK")
    with mlflow.start_run(run_name="fase3_shap_xai"):
        mlflow.log_param("metodo_xai","SHAP GradientExplainer"); mlflow.log_param("framework","PyTorch")
        top3=pd.Series(ms,index=FEATURES).nlargest(3)
        mlflow.log_param("top1_feature",top3.index[0]); mlflow.log_param("top2_feature",top3.index[1])
        [mlflow.log_artifact(a) for a in arts]
        [mlflow.log_metric(f"shap_{f}",round(float(v),6)) for f,v in zip(FEATURES,ms)]
    print(f"\nFase 3 concluida! {len(arts)} artefatos | Top: {list(top3.index)}")

if __name__=="__main__":
    pipeline_explicar()
