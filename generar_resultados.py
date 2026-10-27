"""Genera resultados oficiales reproducibles y archivos para el informe."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from red_xor import RedXOR, X, Y
out=Path(__file__).resolve().parent/'resultados';out.mkdir(exist_ok=True)
experiments={}
for lr in [0.1,1.0]:
    model=RedXOR(seed=17).train(lr=lr,epochs=10000)
    experiments[lr]=model
    fig,ax=plt.subplots(figsize=(7.2,3.6));ax.plot(np.arange(1,10001),model.historial)
    ax.set(xlabel='Épocas',ylabel='Pérdida BCE',title=f'Entrenamiento XOR — tasa de aprendizaje η={lr}')
    ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/f'loss_lr_{str(lr).replace(".","_")}.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7.4,3.6))
for lr,net in experiments.items():ax.plot(np.arange(1,10001),net.historial,label=f'η = {lr}')
ax.set(xlabel='Épocas',ylabel='Pérdida BCE',title='Comparación de convergencia: red 2–2–1')
ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(out/'comparacion_loss.png',dpi=180);plt.close(fig)
with open(out/'perdidas_experimentos.csv','w',newline='',encoding='utf8') as f:
    writer=csv.writer(f);writer.writerow(['epoca','loss_lr_0_1','loss_lr_1_0'])
    for i in range(10000):writer.writerow([i+1,experiments[0.1].historial[i],experiments[1.0].historial[i]])
with open(out/'resumen_experimentos.txt','w',encoding='utf8') as f:
    for lr,net in experiments.items():
        m=net.metrics();msg=f'eta={lr}: loss inicial tras época 1={net.historial[0]:.9f}, pérdida final={m["loss"]:.9f}, exactitud={m["accuracy"]:.0%}, probabilidades={np.round(m["prob"],6)}\n'
        f.write(msg);print(msg.strip())
