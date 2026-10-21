from datasets.datapriority import small_data, medium_data, large_data

import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_small():
    global vectorizer, model
    vectorizer, model = train_model(small_data)
def train_medium():
    global vectorizer, model
    vectorizer, model = train_model(medium_data)
def train_large():
    global vectorizer, model
    vectorizer, model = train_model(large_data)

def train_model(data):
    vectorizer = TfidfVectorizer()
    model = LogisticRegression()
    
    texts = []
    labels = []

    for text, label in data:
        texts.append(text)
        labels.append(label)


    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size = 0.2, random_state = 42, stratify=labels)

    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)

    model.fit(X_train, labels_train)
    predictions = model.predict(X_test)

    print(f"\nОбучение на {len(data)} сообщениях")
    print(classification_report(labels_test, predictions, zero_division=0))
    return vectorizer, model

def classify_message():
    message = message_entry.get().strip()
    if not message:
        result_label.config(text = "Сначала введите сообщение")
        return

    message_X = vectorizer.transform([message])
    prediction = model.predict(message_X)

    categories = {"urgent": "Срочно", "later": "Можно позже", "no_reply": "Отвечать не нужно"}

    result_label.config(text = f"Приоритет: {categories[prediction[0]]}")



window = tk.Tk()
window.title("Prioritityn")
window.geometry("600x300")

message_label = tk.Label(window, text="Введите сообщение:")
message_label.pack(pady=10)

message_entry = tk.Entry(
    window,
    width=50,
    bg="#e0e0e0", 
    fg="#202020",  
    insertbackground="#202020" 
)
message_entry.pack(pady=5)

classify_button = tk.Button(
    window,
    text="Определить",
    command=classify_message
)
classify_button.pack(pady=10)

buttons_frame = tk.Frame(window) 
buttons_frame.pack(pady=10)

button_5 = tk.Button(
    buttons_frame, text="Точность (5)", command=train_small
)
button_5.pack(side="left", padx=5)

button_15 = tk.Button(
    buttons_frame, text="Точность (15)", command=train_medium
)
button_15.pack(side="left", padx=5)

button_45 = tk.Button(
    buttons_frame, text="Точность (45)", command=train_large
)
button_45.pack(side="left", padx=5)

result_label = tk.Label(window, text="Здесь появится категория")
result_label.pack(pady=10)


window.mainloop()

