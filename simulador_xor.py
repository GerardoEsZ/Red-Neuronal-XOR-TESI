"""Simulador visual XOR: ejecutar con 'python simulador_xor.py'."""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from red_xor import RedXOR, X, Y

BG='#eff4fa'; INK='#172a45'; BLUE='#285b9c'; GOOD='#14825d'
class Simulador:
    def __init__(self,root):
        self.root=root; root.title('Laboratorio visual XOR | Redes neuronales artificiales');root.geometry('1270x820');root.minsize(1020,700)
        self.net=RedXOR();self.comparacion={};self.fila=tk.IntVar(value=1);self.lr=tk.StringVar(value='1.0');self.epocas=tk.StringVar(value='10000');self.status=tk.StringVar(value='Inicializa la red y explora cada etapa.')
        self.nb=ttk.Notebook(root); self.nb.pack(fill='both',expand=True,padx=12,pady=10)
        self.main=ttk.Frame(self.nb); self.tests=ttk.Frame(self.nb); self.theory=ttk.Frame(self.nb)
        self.nb.add(self.main,text='  Laboratorio interactivo  ');self.nb.add(self.tests,text='  Experimentos y gráficas  ');self.nb.add(self.theory,text='  Guía para explicar al profesor  ')
        controls=ttk.Frame(self.main);controls.pack(fill='x',pady=8)
        ttk.Label(controls,text='Tasa η:').pack(side='left',padx=4);ttk.Entry(controls,textvariable=self.lr,width=7).pack(side='left')
        ttk.Label(controls,text='Épocas:').pack(side='left',padx=4);ttk.Entry(controls,textvariable=self.epocas,width=9).pack(side='left')
        for txt,fn in [('Reiniciar pesos',self.reset),('Entrenar 1 época',self.one),('Entrenar completo',self.full),('Guardar gráfica',self.saveplot)]:ttk.Button(controls,text=txt,command=fn).pack(side='left',padx=5)
        body=ttk.PanedWindow(self.main,orient='horizontal');body.pack(fill='both',expand=True)
        left=ttk.Frame(body);right=ttk.Frame(body);body.add(left,weight=3);body.add(right,weight=2)
        self.diagram=tk.Canvas(left,bg='white',height=375,highlightthickness=1,highlightbackground='#c8d8e8');self.diagram.pack(fill='x',padx=8,pady=6)
        pick=ttk.Frame(left);pick.pack(fill='x',padx=8)
        ttk.Label(pick,text='Inspeccionar entrada XOR:').pack(side='left')
        for i,case in enumerate(['(0,0) → 0','(0,1) → 1','(1,0) → 1','(1,1) → 0']):ttk.Radiobutton(pick,text=case,variable=self.fila,value=i,command=self.update).pack(side='left',padx=7)
        self.pred=tk.Text(left,height=10,wrap='word',font=('Consolas',10),bg='#f7faff');self.pred.pack(fill='both',expand=True,padx=8,pady=8)
        self.details=tk.Text(right,wrap='word',font=('Consolas',10),bg='#f7faff');self.details.pack(fill='both',expand=True,padx=6,pady=6)
        ttk.Label(self.main,textvariable=self.status,font=('Arial',10,'bold')).pack(anchor='w',padx=10,pady=4)
        self.setup_tests();self.setup_theory();self.update()
    def parse(self):
        try:
            lr=float(self.lr.get()); n=int(self.epocas.get())
            if not (0<lr<=3 and 1<=n<=50000):raise ValueError()
            return lr,n
        except ValueError:
            messagebox.showerror('Dato inválido','Usa tasa entre 0 y 3 (por ejemplo 1.0) y épocas entre 1 y 50000.');return None
    def reset(self):self.net=RedXOR();self.status.set('Pesos reiniciados con semilla 17 para reproducibilidad.');self.update()
    def one(self):
        pars=self.parse()
        if pars:
            info=self.net.step(pars[0]);self.status.set(f'Época {self.net.epoch}: pérdida {info["antes"]:.6f} → {info["despues"]:.6f}; pesos ajustados con retropropagación.');self.update(info)
    def full(self):
        pars=self.parse()
        if not pars:return
        lr,n=pars
        self.root.config(cursor='watch');self.root.update_idletasks()
        try:
            self.net.train(lr,n)
            self.status.set(f'Entrenamiento terminado: {self.net.epoch} épocas acumuladas, error {self.net.loss():.6f}, exactitud {self.net.metrics()["accuracy"]:.0%}.')
            self.update();self.draw_history()
        finally:self.root.config(cursor='')
    def update(self,info=None):
        f=self.net.forward(X[self.fila.get()]);i=self.fila.get();a=f['a1'][0];p=float(f['a2'][0,0]);tar=int(Y[i,0]);decision=int(p>=.5)
        c=self.diagram;c.delete('all'); w=max(c.winfo_width(),750);pos=[(w*.13,125),(w*.13,275),(w*.50,125),(w*.50,275),(w*.86,200)]
        for i1,j1 in [(0,2),(0,3),(1,2),(1,3),(2,4),(3,4)]:
            weight=float(self.net.W1[i1,j1-2]) if i1<2 else float(self.net.W2[i1-2,0]); x1,y1=pos[i1];x2,y2=pos[j1]
            c.create_line(x1+25,y1,x2-28,y2,fill='#8aa0b9',width=2,arrow=tk.LAST)
            mx=x1+(x2-x1)*.52;my=y1+(y2-y1)*.52
            c.create_text(mx,my-12,text=f'w={weight:+.2f}',fill='#324b6b',font=('Consolas',9))
        names=[f'x₁={int(f["x"][0,0])}',f'x₂={int(f["x"][0,1])}',f'h₁={a[0]:.3f}',f'h₂={a[1]:.3f}',f'ŷ={p:.3f}']
        for j,((x,y),label) in enumerate(zip(pos,names)):
            c.create_oval(x-42,y-42,x+42,y+42,fill=('#dcf4e9' if j==4 else '#daebfd'),outline=GOOD if j==4 else BLUE,width=2)
            c.create_text(x,y,text=label,font=('Arial',12,'bold'),fill=INK)
        for x,txt in [(w*.13,'CAPA DE ENTRADA'),(w*.50,'CAPA OCULTA'),(w*.86,'SALIDA')]:c.create_text(x,26,text=txt,font=('Arial',12,'bold'),fill=BLUE)
        c.create_text(w*.52,347,text=f'Sesgos: capa oculta [{self.net.b1[0,0]:+.3f}, {self.net.b1[0,1]:+.3f}] | salida {self.net.b2[0,0]:+.3f}',font=('Consolas',11),fill=INK)
        self.details.delete('1.0',tk.END)
        explanation=f'''PASO 1 · ENTRADA SELECCIONADA
x = {f["x"][0]} | esperado = {tar}

PASO 2 · SUMA PONDERADA EN CAPA OCULTA
z₁ = x · W₁ + b₁ = {f["z1"][0]}
W₁ =\n{self.net.W1}
b₁ = {self.net.b1[0]}

PASO 3 · ACTIVACIÓN SIGMOIDE
σ(z) = 1/(1+exp(-z))
h = σ(z₁) = {a}

PASO 4 · SALIDA
z₂ = h · W₂ + b₂ = {f["z2"][0,0]:.6f}
W₂ = {self.net.W2.reshape(-1)}
b₂ = {self.net.b2[0,0]:.6f}
ŷ = σ(z₂) = {p:.6f}
Clase (umbral 0.5) = {decision}

PASO 5 · ERROR
Pérdida global BCE (4 casos) = {self.net.loss():.6f}
Épocas realizadas = {self.net.epoch}
'''
        if info:explanation+='''\nPASO 6 · RETROPROPAGACIÓN (ÚLTIMA ÉPOCA)
Gradiente W₂:\n{}\nGradiente b₂: {}\nGradiente W₁:\n{}\nGradiente b₁: {}\nRegla: W_nuevo = W_anterior − η·gradiente\n'''.format(info['grad_W2'],info['grad_b2'],info['grad_W1'],info['grad_b1'])
        self.details.insert('1.0',explanation)
        self.pred.delete('1.0',tk.END)
        rows=['ENTRADAS      OBJETIVO   PROBABILIDAD   CLASE   ESTADO','-'*60]
        for j,(xy,t,q) in enumerate(zip(X,Y.reshape(-1),self.net.metrics()['prob'])):
            ans=int(q>=.5);rows.append(f'{str(tuple(map(int,xy))):<13}{int(t):^9}   {q:>9.5f}       {ans:^3}    {"✓" if ans==t else "✗"}')
        rows.append(f'\nExactitud: {self.net.metrics()["accuracy"]:.0%} | Pérdida: {self.net.loss():.6f}')
        self.pred.insert('1.0','\n'.join(rows));self.draw_history()
    def setup_tests(self):
        top=ttk.Frame(self.tests);top.pack(fill='x',padx=15,pady=10)
        ttk.Label(top,text='Comparación controlada: mismas semillas, arquitectura y 10,000 épocas.',font=('Arial',11,'bold')).pack(side='left')
        ttk.Button(top,text='Ejecutar experimentos 0.1 vs 1.0',command=self.run_tests).pack(side='right')
        self.fig=Figure(figsize=(10,4.8),dpi=95);self.ax=self.fig.add_subplot(111);self.plotcanvas=FigureCanvasTkAgg(self.fig,master=self.tests);self.plotcanvas.get_tk_widget().pack(fill='both',expand=True,padx=18,pady=8)
        self.report=tk.Text(self.tests,height=8,font=('Consolas',10),wrap='word');self.report.pack(fill='x',padx=18,pady=8)
        bottom=ttk.Frame(self.tests);bottom.pack(fill='x',padx=16,pady=5)
        ttk.Button(bottom,text='Exportar PNG y CSV a resultados/',command=self.export).pack(side='left')
        self.draw_history()
    def draw_history(self):
        if not hasattr(self,'ax'):return
        self.ax.clear()
        if self.comparacion:
            for rate,values in self.comparacion.items():self.ax.plot(np.arange(1,len(values)+1),values,label=f'η={rate}')
        elif self.net.historial:self.ax.plot(np.arange(1,len(self.net.historial)+1),self.net.historial,label='Entrenamiento actual')
        self.ax.set_title('Evolución de la pérdida de la red XOR');self.ax.set_xlabel('Épocas');self.ax.set_ylabel('Entropía cruzada binaria (Loss)');self.ax.grid(alpha=.25);self.ax.legend() if self.ax.lines else None
        self.fig.tight_layout();self.plotcanvas.draw_idle()
    def run_tests(self):
        self.root.config(cursor='watch');self.root.update_idletasks()
        try:
            self.comparacion={};lines=['Experimento | Pérdida final | Exactitud | Predicciones (orden XOR)','-'*85]
            for rate in (.1,1.0):
                net=RedXOR().train(rate,10000);self.comparacion[rate]=net.historial
                result=net.metrics();lines.append(f'η={rate:<4}      {result["loss"]:.7f}        {result["accuracy"]:.0%}       {np.array2string(result["prob"],precision=4)}')
            lines.append('\nMisma semilla 17, épocas y arquitectura. Solo varía η.')
            self.report.delete('1.0',tk.END);self.report.insert('1.0','\n'.join(lines));self.draw_history()
        finally:self.root.config(cursor='')
    def export(self):
        if not self.comparacion:self.run_tests()
        from pathlib import Path
        folder=Path(__file__).resolve().parent/'resultados';folder.mkdir(exist_ok=True)
        self.fig.savefig(folder/'comparacion_loss.png',dpi=180,bbox_inches='tight')
        rows=np.column_stack([np.arange(1,10001),self.comparacion[.1],self.comparacion[1.0]])
        np.savetxt(folder/'perdidas_experimentos.csv',rows,delimiter=',',header='epoca,loss_lr_0_1,loss_lr_1_0',comments='')
        messagebox.showinfo('Exportación realizada',f'Guardados comparacion_loss.png y perdidas_experimentos.csv en:\n{folder}')
    def saveplot(self):
        path=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('Imagen PNG','*.png')]);
        if path:self.fig.savefig(path,dpi=180,bbox_inches='tight')
    def setup_theory(self):
        msg='''GUÍA DE DEMOSTRACIÓN PARA EL PROFESOR\n\n1. XOR: 00→0, 01→1, 10→1, 11→0. No puede clasificarse con una sola recta.\n\n2. Se muestra la arquitectura 2–2–1: 2 entradas, 2 neuronas ocultas y 1 salida. Los números junto a las conexiones son pesos entrenables.\n\n3. Puede seleccionar cualquiera de las cuatro entradas; muestra z₁ = xW₁+b₁, activación sigmoide h, z₂ = hW₂+b₂ y salida ŷ.\n\n4. Puede pulsar «Entrenar 1 época». El programa calcula los cuatro ejemplos (lote completo), la entropía cruzada y los gradientes con la regla de la cadena. Actualiza W y b con W ← W − η∇L.\n\n5. Al pulsar «Entrenar completo». Muestra cómo la pérdida disminuye y las probabilidades se aproximan a los objetivos. La clase se decide aplicando umbral 0.5.\n\n6. En «Experimentos y gráficas», puede ejecutar η=0.1 frente a η=1.0 durante 10,000 épocas con la misma inicialización; se puede ver la influencia de η sobre la velocidad de convergencia.\n\n7. Indica que 100% de acierto sobre los cuatro ejemplos pero esto NO significa generalización, porque XOR tiene un conjunto diminuto y no existe conjunto de prueba independiente. No se puede concluir empíricamente sobreajuste con estos datos.\n\nPREGUNTAS FRECUENTES\n• ¿Qué aprende? Los pesos y sesgos de ambas capas.\n• ¿Qué función de pérdida? Entropía cruzada binaria promedio (BCE).\n• ¿Qué cambia en retropropagación? Se calculan derivadas y se ajustan parámetros de todas las capas.\n• ¿Por qué no una sola neurona? Porque XOR no es linealmente separable.\n• ¿Por qué usar semilla fija? Para reproducir y comparar experimentos justamente.\n'''
        t=tk.Text(self.theory,wrap='word',font=('Arial',12),padx=24,pady=22);t.pack(fill='both',expand=True);t.insert('1.0',msg);t.config(state='disabled')

if __name__=='__main__':
    root=tk.Tk();sty=ttk.Style(root)
    if 'clam' in sty.theme_names():sty.theme_use('clam')
    app=Simulador(root);root.mainloop()
