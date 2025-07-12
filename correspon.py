import pandas as pd
import numpy as np
import os
import prince  # Correspondence Analysisライブラリ
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
#=============================================
# 調査の車両数
#=============================================
num_car = 3
#=============================================
# Inputファイル情報
#=============================================
INPUT_DNAME = "Car_サンプルデータ.csv"
INPUT_folder = "2_data"        
#=============================================
# Outputファイル情報
#=============================================
OUTPUT_DNAME = "1_平均.csv"
OUTPUT_DIS = "2_行列入れ替え.csv"
OUTPUT_AVE = "3_コレポンデータ.csv"
OUTPUT_folder = "3_output"
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
# Inputデータ読み込む
#=============================================
df = pd.read_csv(input_dpath,encoding='shift-JIS')
#=============================================
# 列名をすべて取得
#=============================================
columns= list(df.columns)

print(", ".join(columns))

#=============================================
# Car列のデータの型を指定
#=============================================
df['Car'] = df['Car'].astype(str)
df['Car.1'] = df['Car.1'].astype(str)
df['Car.2'] = df['Car.2'].astype(str)
#=============================================
# # すべての "Car" で始まる列のインデックスを取得
#=============================================
car_indices = [i for i, col in enumerate(columns) if col.startswith("Car")]
#=============================================
# # データフレームの列数を末尾インデックスとして追加
#=============================================
car_indices.append(len(columns))
#=============================================
# "Car"で始まる列ごとに、範囲を決めて処理する
#=============================================
columns_trimmed_list = []

for idx in range(len(car_indices) - 1):
    start = car_indices[idx]
    end = car_indices[idx + 1]

    # ID列も含める（0番目）
    trimmed = [columns[0]] + columns[start:end]
    columns_trimmed_list.append(trimmed)
#=============================================
# "Car"の列だけを個別のリストにする
#=============================================
columns_trimmed_1 = columns_trimmed_list[0]
columns_trimmed_2 = columns_trimmed_list[1]
columns_trimmed_3 = columns_trimmed_list[2]
# 結果を確認
print(columns_trimmed_1)
print(columns_trimmed_2)
print(columns_trimmed_3)

#=============================================
# まずは、車両数分それぞれのデータフレームに分割
#=============================================
df1 = df[columns_trimmed_1].copy()
df2 = df[columns_trimmed_2].copy()
df3 = df[columns_trimmed_3].copy()
print(df1.head(3))
print(df2.head(3))
print(df3.head(3))
 
#=============================================
# まずは、車両数分それぞれのデータフレームに分割
#=============================================
df1 = df[columns_trimmed_1].copy()
df2 = df[columns_trimmed_2].copy()
df3 = df[columns_trimmed_3].copy()

#=============================================
# カラム名を統一（df2、df3の列名をdf1に合わせる）
#=============================================
df2.columns = columns_trimmed_1
df3.columns = columns_trimmed_1

#=============================================
# 縦に連結
#=============================================
df_long = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
#=============================================
#　データ確認用に出力
#=============================================
df_long.to_csv("1.csv" ,encoding='cp932',index =False)
df_long.head(3)

#=============================================
# イメージワードの開始列を指定（ID列とCar列以降の列）
#=============================================
image_cols = df_long.columns[2:]

#=============================================
# NaNを空文字に置換
#=============================================
df_long[image_cols] = df_long[image_cols].fillna('')

#=============================================
# # "1"（文字）または1（数値）を1に、それ以外は0に変換
#=============================================
df_long[image_cols] = df_long[image_cols].applymap(lambda x: 1 if x == 1 or x == '1' else 0)

df_long[image_cols] .head(3)

#=============================================
# Carごとにイメージワードの平均値を算出
#=============================================
df_grouped = df_long.groupby('Car')[image_cols].mean()
df_grouped

#=============================================
# Carごとにイメージワードの平均値を算出
#=============================================
df_grouped = df_long.groupby('Car')[image_cols].mean()

#=============================================
# インデックスを列に変換
#=============================================
df_grouped = df_grouped.reset_index()

#=============================================
# 集計結果をCSVファイルに出力
#=============================================            
df_grouped.to_csv(output_dpath + "\\" + OUTPUT_DNAME ,encoding='cp932',index =False)
df_grouped

#=============================================
# 行列入れ替え
#=============================================  

data = df_grouped.T
data

df_transposed = df_grouped.T.reset_index()
df_transposed

#=============================================
# 先頭行を車両名にする
#=============================================   
data.columns = data.iloc[0]

#=============================================
# 数値のカラム名になってる先頭行を削除
#=============================================  
data_a = data.drop(data.index[0])

#=============================================
# csvファイルに出力
#=============================================            
data_a.to_csv(output_dpath + "\\" + OUTPUT_DIS ,encoding='cp932',index =False)
data_a

#=============================================  
# data_aの各列をリスト化して辞書に変換
#=============================================  
data_b = {col: data_a[col].tolist() for col in data_a.columns}

#=============================================  
# 元のdata_aのインデックスもリストで取得
#============================================= 
index = data_a.index.tolist()

#=============================================  
#辞書とインデックスを使い、同じ内容の新しいDataFrame  を作成
#============================================= 
data_c = pd.DataFrame(data_b, index=index)
data_c

#=============================================  
# コレスポンデンス分析の実行準備
#============================================= 
ca = prince.CA(
    n_components=2,
    n_iter=10,
    copy=True,
    check_input=True,
    engine='sklearn'
)
#=============================================  
# 分析の実行
#============================================= 
ca = ca.fit(data_c)
#============================================= 
# 行（イメージワード）と列（車種）の座標を取得
#============================================= 
row_coords = ca.row_coordinates(data_c)
col_coords = ca.column_coordinates(data_c)


#============================================= 
# 日本語フォントを指定(私のPCの場合）
#============================================= 
font_path = r"C:\WINDOWS\Fonts\YuGothR.ttc"
prop = fm.FontProperties(fname=font_path)

#============================================= 
# プロットの準備（図のサイズ指定）
#============================================= 
fig, ax = plt.subplots(figsize=(8, 6))

#============================================= 
# 行（イメージワード）の散布図を青色で描画
#============================================= 
ax.scatter(row_coords[0], row_coords[1], color='blue', label='Impression')
for i, txt in enumerate(row_coords.index):
    ax.annotate(
        txt,
        (row_coords.iloc[i, 0], row_coords.iloc[i, 1]),
        color='blue',
        fontproperties=prop
    )
#============================================= 
# 列（車種）の散布図を赤色で描画
#============================================= 
ax.scatter(col_coords[0], col_coords[1], color='red', label='Car')
for i, txt in enumerate(col_coords.index):
    ax.annotate(
        txt,
        (col_coords.iloc[i, 0], col_coords.iloc[i, 1]),
        color='red',
        fontproperties=prop
    )
#============================================= 
# x軸・y軸のゼロラインを灰色で描画
#============================================= 
ax.axhline(0, color='grey', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5)

#============================================= 
# タイトルもフォント指定
#============================================= 
ax.set_title(
    'Correspondence Analysis: Car Brands and Impressions',
    fontproperties=prop
)
# 凡例を表示
ax.legend()
# プロットを画面に表示

plt.show()

#============================================= 
# 車種と印象語のラベルに区別用の列を追加
#============================================= 
col_coords['type'] = 'Car'
row_coords['type'] = 'Impression'

#============================================= 
# 結合
#============================================= 
combined = pd.concat([col_coords, row_coords])

#============================================= 
# 出力
#============================================= 
combined.to_csv(output_dpath + "\\" + OUTPUT_AVE ,encoding='cp932')

# 結果を表示
print(combined)

#=============================================
# 保存フォルダ開く
#=============================================
os.startfile(os.path.realpath(output_dpath) + "\\")
#=============================================
# 保存したファイル開く
#=============================================
os.startfile(os.path.realpath(output_dpath) + "\\" + OUTPUT_AVE)