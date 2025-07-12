import pandas as pd
import numpy as np

import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus
from IPython.display import Image
import matplotlib.pyplot as plt

#=============================================
# Inputファイル情報
#=============================================
INPUT_folder = '2_data'        
INPUT_DNAME = '11.csv'
#=============================================
# Outputファイル情報
#=============================================
OUTPUT_folder = '3_output'
#=============================================
# カレントパス
#=============================================
current_dpath = os.getcwd()
#=============================================
# パレントパス
#=============================================
parent_dpath =os.path.sep.join(current_dpath.split(os.path.sep)[:-1])
#=============================================
# Inputデータファイル Path
#=============================================
input_dpath =os.path.sep.join([parent_dpath + '\\' + INPUT_folder,INPUT_DNAME])

#=============================================
# Outputデータファイル Path
#=============================================
output_dpath =parent_dpath + '\\' + OUTPUT_folder

#=============================================
# サンプルデータ読み込む
#=============================================
df = pd.read_csv(input_dpath,encoding='shift-JIS')
df = pd.DataFrame(df)
#=============================================
#説明変数 だけにする
#=============================================
X_df = df.drop(['ID','car', 'cluster'], axis=1)

#=============================================
#目的変数 
#=============================================
y_df = df['car']

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state=0)

print(len(X_train))
print(len(X_test))

# 決定木モデルを構築するクラスを初期化
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# 決定木モデルを生成
model = clf.fit(X_train, y_train)

# 訓練・テストそれぞれの正解率を算出
print('正解率(train):{:.3f}'.format(model.score(X_train, y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test, y_test)))

def add_class_legend(legend_text, x=0.02, y=0.98):
    """
    決定木の凡例テキストを現在のFigureに追加する。
    """
    plt.gcf().text(
        x, y,
        legend_text,
        fontsize=13,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.5",
            fc="lightyellow",
            ec="gray",
            lw=1
        )
    )


# 決定木を描画
model.fit(X_df, y_df)

# クラスのインデックス対応表を作成
legend_text = "Class index mapping:\n"
for idx, class_label in enumerate(model.classes_):
    legend_text += f"{idx}: {class_label}\n"
    

# 比率なし
plt.figure(figsize=(20,10))
tree.plot_tree(
    model,
    feature_names=X_df.columns,
    class_names=model.classes_.astype(str),
    filled=True,
    max_depth=6
)

add_class_legend(legend_text)  # ←ここで関数を呼ぶ

plt.savefig("decision_tree_N.png", dpi=300)
plt.show()

# 比率あり
plt.figure(figsize=(20,10))
tree.plot_tree(
    model,
    feature_names=X_df.columns,
    class_names=model.classes_.astype(str),
    filled=True,
    max_depth=6,
    proportion=True
)

add_class_legend(legend_text)  # ←再利用できる

plt.savefig("decision_tree.png", dpi=300)
plt.show()
