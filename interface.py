import tkinter as tk
import tkinter as ttk
from tkinter.constants import CENTER, E, END, N, S, W
from tkinter.ttk import Frame, Notebook
from tkinter import scrolledtext, filedialog
from nltk.translate.bleu_score import sentence_bleu

def open():
    filename = filedialog.askopenfile(title='Open Text file', filetypes=(('Text Files','*.txt'),))
    #path.insert(END, filename)
    file = open(filename) # or tf = open(tf, 'r')
    data = file.read()
    txtarea.insert(END, data)
    file.close()

def calc_similarity():
    desc1 = text_area.get("1.0",END)
    reference = [desc1.split()]
    desc2 = text2_area.get("1.0",END)
    candidate = desc2.split()
    similarity = sentence_bleu(reference, candidate) * 100
    print(similarity)
    #similarity = str(round(similarity,2))
    result = tk.Label(root, text= 'Similarity is:' + str(similarity), font=('Times New Roman',14), borderwidth=5)
    result.grid(row=6, column=0)

root = tk.Tk()
root.title('Similarity Computation')
root.geometry('450x600')
title = tk.Label(root, text='Computer the similiarity between two descriptions:', font=('Times New Roman', 15))
title.grid(row=0, column=0, padx=20, pady=10)

#----------------------------------------Course 1-------------------------------------------------------

notebook = Notebook(root)
notebook.grid(row=1, column=0, padx=20, pady=10)

txt_frame = Frame(notebook)
text_area = scrolledtext.ScrolledText(txt_frame, wrap = tk.WORD, width = 40, height = 8, font = ("Times New Roman",11))
text_area.grid(row=2, column=0, padx=20, pady=10)

file_frame = Frame(notebook)
file_button = tk.Button(file_frame, text='Open File', command= open)
file_button.grid(row=3, column=0, padx= 20, pady=10)

txtarea = tk.Text(file_frame, width=40, height=8)
txtarea.grid(row=4, column=0, padx=20, pady=10)
# path = tk.Entry(file_frame)
# path.grid(row=4, column=0, padx=20, pady=10)


notebook.add(txt_frame ,text='Enter text')
notebook.add(file_frame, text='Open File')

#----------------------------------------Course 2---------------------------------------------cd ----------
notebook2 = Notebook(root)
notebook2.grid(row=4, column=0, padx=20, pady=10)

txt2_frame = Frame(notebook2)
text2_area = scrolledtext.ScrolledText(txt2_frame, wrap = tk.WORD, width = 40, height = 8, font = ("Times New Roman",11))
text2_area.grid(row=5, column=0, padx=20, pady=10)

file2_frame = Frame(notebook2)
file2_button = tk.Button(file2_frame, text='Open File', command= open)
file2_button.grid(row=4, column=0, padx= 20, pady=10)

txtarea2 = tk.Text(file2_frame, width=40, height=8)
txtarea2.grid(row=5, column=0,padx=20, pady=10)

notebook2.add(txt2_frame ,text='Enter text')
notebook2.add(file2_frame, text='Open File')

#----------------------------------------Calculate-------------------------------------------------------
calc_button = tk.Button(root, text = "Calculate Similarity", command = calc_similarity)
#calc_button.place(relx=0.5, rely=0.5, anchor=CENTER)
calc_button.grid(row=6, column=0, padx=20, pady=10,)

root.mainloop()