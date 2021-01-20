import itertools
import pydotplus
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('dtset_einst.csv', delimiter=';')

print(df.head(5))

count_row = df.shape[0]
count_col = df.shape[1]

print(count_row)
print(count_col)

df = df.dropna()

print(df.head(5))

print('Quant de campos(col): ', df.shape[1])
print('Tot registros:', df.shape[0])

print ('Tot registros negativos: ', df[df['SARS-Cov-2 exam result'] =='negativo'].shape[0])
print ('Tot registros positivos: ', df[df['SARS-Cov-2 exam result'] =='positivo'].shape[0])

Y = df['SARS-Cov-2 exam result'].values
print(Y)

X = df[['Hemoglobina', 'leucócitos ', 'Basophils','Proteina C reativa mg/dL']].values

print(X)


X_tre, X_tes, Y_tre, Y_tes = train_test_split(X, Y, test_size=0.2, random_state=3)


algortimo_arvore = DecisionTreeClassifier(criterion='entropy', max_depth=5)

modelo = algortimo_arvore.fit(X_tre, Y_tre)

print (modelo.feature_importances_)

n_features = ['Hemoglobina', 'leucócitos ', 'Basophils', 'Proteina C reativa mg/dL']
n_classes = modelo.classes_

dot_dat = StringIO()

export_graphviz(modelo, out_file=dot_dat, filled=True, feature_names=n_features, class_names=n_classes, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_dat.getvalue())
Image(graph.create_png())
graph.write_png("arvore.png")
Image('arvore.png')

importance = modelo.feature_importances_
ind = np.argsort(importance)[::-1]
print("--- Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, ind[f], importance[ind[f]]))
f, ax = plt.subplots(figsize=(11, 9))
plt.title("Feature ranking", fontsize = 20)
plt.bar(range(X.shape[1]), importance[ind],
        color="b",
        align="center")
plt.xticks(range(X.shape[1]), ind)
plt.xlim([-1, X.shape[1]])
plt.ylabel("importance", fontsize = 18)
plt.xlabel("index of the feature", fontsize = 18)
plt.show()

Y_pred = modelo.predict(X_tes)

print("---- ACURÁCIA DA ÁRVORE: ", accuracy_score(Y_tes, Y_pred))
print (classification_report(Y_tes, Y_pred))


def plot_conf_matrix(cm, classes,
                     normalize=False,
                     title='Conf matrix',
                     cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Mat Conf Normalizada")
    else:
        print('Mat Conf sem normalizacão ')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="branco" if cm[i, j] > thresh else "preto")

    plt.tight_layout()
    plt.ylabel('Rótulo true ---')
    plt.xlabel('Rótulo prevista ---')

mat_conf = confusion_matrix(Y_tes, Y_pred)
plt.figure()
plot_conf_matrix(mat_conf, classes=n_classes,
                 title='Mat Conf')