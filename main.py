import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Data
data = [
    ["Persita", 10, 3, 1, 1, 6, 2, "babak selanjutnya"],
    ["Dewa United", 10, 3, 1, 1, 9, 3, "babak selanjutnya"],
    ["Madura United", 10, 3, 1, 1, 11, 2, "babak selanjutnya"],
    ["Bali United", 9, 3, 0, 2, 9, 2, "babak selanjutnya"],
    ["Persija Jakarta", 8, 2, 2, 1, 6, 3, "babak selanjutnya"],
    ["RANS Nusantara", 8, 2, 2, 1, 6, 1, "babak selanjutnya"],
    ["PSM Makasar", 8, 2, 2, 1, 7, 2, "babak selanjutnya"],
    ["Borneo FC", 8, 2, 2, 1, 7, 2, "babak selanjutnya"],
    ["PSIS", 8, 2, 2, 1, 7, 2, "babak selanjutnya"],
    ["Barito Putera", 7, 2, 1, 2, 8, 2, "babak selanjutnya"],
    ["PSS", 6, 1, 3, 1, 6, -1, "gugur"],
    ["Persib Bandung", 6, 1, 3, 1, 10, -1, "gugur"],
    ["Persebaya", 5, 1, 2, 2, 6, -2, "gugur"],
    ["Persikabo 1973", 5, 1, 2, 2, 6, 0, "gugur"],
    ["Persis", 5, 1, 2, 2, 10, -1, "gugur"],
    ["Persik", 4, 1, 1, 3, 9, -2, "gugur"],
    ["Arema FC", 2, 0, 2, 3, 7, -6, "gugur"],
    ["Bayangkara", 1, 0, 1, 4, 4, -8, "gugur"]
]

# Membuat DataFrame
columns = ["Klub", "Poin", "Menang", "Seri", "Kalah", "Goal", "SG", "kelas"]
df = pd.DataFrame(data, columns=columns)

# Mengubah kelas menjadi angka
df['kelas'] = df['kelas'].map({'babak selanjutnya': 1, 'gugur': 0})

# Memisahkan fitur dan target
X = df.drop(['Klub', 'kelas'], axis=1)
y = df['kelas']

# Membuat model Decision Tree
model = DecisionTreeClassifier(random_state=5)

# Melatih model
model.fit(X, y)

# Prediksi juara
prediksi_juara = model.predict([[10, 3, 1, 1, 6, 2]])
if prediksi_juara[0] == 1:
    print("Final")
else:
    print("Gagal")

# Prediksi klub yang menang
poin_max = df[df['kelas'] == 1]['Poin'].max()
klub_pemenang = df[(df['Poin'] == poin_max) & (df['kelas'] == 1)]['Klub'].values[0]
print("Klub yang diprediksi akan menang:", klub_pemenang)