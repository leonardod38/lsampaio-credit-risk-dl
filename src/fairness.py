"""
fairness.py — Fase 4: Analise de fairness por grupo demografico (PyTorch).
Autor: Leonardo Sampaio
"""
import sys, mlflow, numpy as np, pandas as pd
import matplotlib.pyplot as plt, torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=ROOT/"data"; MODELS_DIR=ROOT/"models"
FIGS_DIR=ROOT/"reports"/"figures"
TRACKING_URI="http://127.0.0.1:5000"
EXPERIMENT="classificacao-risco-credito"
CLASSES=["baixo","medio","alto","critico"]
ATRIBS=["genero","faixa_etaria","regiao"]
FEATURES=["score_credito","renda_mensal","divida_total","ratio_divida_renda",
"historico_pagamento","tempo_emprego_anos","num_contas","num_atrasos",
"idade","score_normalizado","genero_enc","regiao_enc","faixa_etaria_enc"]

def carregar_modelo():
    sys.path.insert(0,str(ROOT/"src"))
    from train import CreditRiskNet
    pts=sorted(MODELS_DIR.glob("*.pt"))
    if not pts: print("Execute train.py primeiro."); sys.exit(1)
    m=CreditRiskNet(len(FEATURES))
    m.load_state_dict(torch.load(str(pts[-1]),weights_only=True)); m.eval()
    return m

def metricas_por_grupo(df,y_pred,atrib):
    rows=[]
    for g in df[atrib].unique():
        mask=df[atrib]==g
        yt=df.loc[mask,"label"].values; yp=y_pred[mask]
        if len(yt)<5: continue
        rows.append({"atributo":atrib,"grupo":g,"n":int(mask.sum()),
                     "accuracy":round(accuracy_score(yt,yp),4),
                     "f1_macro":round(f1_score(yt,yp,average="macro",zero_division=0),4)})
    return pd.DataFrame(rows)

def plot_barras(df_m,metrica,titulo,fname):
    FIGS_DIR.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,len(ATRIBS),figsize=(15,5))
    for ax,atr in zip(axes,ATRIBS):
        sub=df_m[df_m["atributo"]==atr]
        if sub.empty: continue
        cores=["#D85A30" if v<sub[metrica].mean() else "#1D9E75" for v in sub[metrica]]
        ax.bar(sub["grupo"],sub[metrica],color=cores,edgecolor="none")
        ax.axhline(sub[metrica].mean(),color="gray",linestyle="--",alpha=0.7,label="Media")
        ax.set_title(atr,fontsize=11); ax.tick_params(axis="x",rotation=30)
        ax.spines[["top","right"]].set_visible(False); ax.legend(fontsize=8)
    plt.suptitle(titulo,fontsize=13,y=1.02); plt.tight_layout()
    p=str(FIGS_DIR/fname); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close(); return p

def pipeline_fairness():
    mlflow.set_tracking_uri(TRACKING_URI); mlflow.set_experiment(EXPERIMENT)
    modelo=carregar_modelo()
    df_te=pd.read_csv(DATA_DIR/"credito_test.csv")
    X_te=torch.FloatTensor(df_te[FEATURES].values)
    y_te=df_te["label"].values
    with torch.no_grad():
        y_pred=modelo(X_te).argmax(1).numpy()
    df_raw=pd.read_csv(DATA_DIR/"credito_raw.csv")
    df_fair=df_raw.sample(len(y_te),random_state=42).reset_index(drop=True)
    df_fair["label"]=y_te; df_fair["pred"]=y_pred
    todos=[]
    for atr in ATRIBS:
        todos.append(metricas_por_grupo(df_fair,y_pred,atr))
    df_todos=pd.concat(todos,ignore_index=True)
    pa=plot_barras(df_todos,"accuracy","Accuracy por Grupo Demografico","fairness_accuracy.png")
    pf=plot_barras(df_todos,"f1_macro","F1 por Grupo Demografico","fairness_f1.png")
    pc=str(FIGS_DIR/"fairness_report.csv"); df_todos.to_csv(pc,index=False)
    acc_g=accuracy_score(y_te,y_pred); f1_g=f1_score(y_te,y_pred,average="macro")
    with mlflow.start_run(run_name="fase4_fairness"):
        mlflow.log_param("atributos_sensiveis",str(ATRIBS))
        mlflow.log_metric("acc_global",round(acc_g,4))
        mlflow.log_metric("f1_global",round(f1_g,4))
        for atr in ATRIBS:
            sub=df_todos[df_todos["atributo"]==atr]
            if not sub.empty:
                d=sub["accuracy"].max()-sub["accuracy"].min()
                mlflow.log_metric(f"disparidade_acc_{atr}",round(float(d),4))
        for p in [pa,pf,pc]: mlflow.log_artifact(p)
    print(f"\nFase 4 concluida!")
    print(f"  Accuracy global: {acc_g:.4f} | F1: {f1_g:.4f}")
    print(f"\nDisparidades:")
    for atr in ATRIBS:
        sub=df_todos[df_todos["atributo"]==atr]
        if not sub.empty:
            print(f"  {atr}: {sub['accuracy'].max()-sub['accuracy'].min():.4f}")

if __name__=="__main__":
    pipeline_fairness()
