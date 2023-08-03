# 特徴抽出

## 2.1 Attentionを使った特徴抽出

**Attentionの特長：**   
画像のどの部分から特徴を抽出するかを動的に決定する。  
**クエリ（Query）** と呼ばれる特徴収集用のベクトルを用意して特徴抽出を行う。  
クエリは、入力画像の情報を使って生成することができるので、画像の色に合わせてクエリを生成することができる。よって、畳み込み演算のようにカーネルを多数用意する必要がない。

- 畳み込みとの違い  
  畳み込み演算を使った特徴抽出では、認識に必要な特徴を正しい領域から抽出できるように事前にカーネルを決定する必要があるため、入力画像が多様である場合、全ての画像の特徴に対応できるように多数のカーネルを用意する必要がある。


**Attentionの計算：**  
### **(1)特徴空間への射影**  
各画素に対して以下の式のように、各画素に対して重み行列Wをかけて画像を特徴空間に射影する

$$ F(x,y) = WI(x,y) ......[2.7] $$

$I(x,y)$ は座標(x,y)におけるRGBの３つのチャネルの値を表すベクトルとし、$F(x,y)$は座標(x,y)における特徴量を表すベクトルとする。

### **（2）正規化**
特徴量とクエリqの内積をとりソフトマックス(softmax)を計算。  
ソフトマックス：入力に対して各成分が正で、成分の合計が1になるように調整する関数  


**計算方法：**  
入力が$x=[x_1, x_2, ... , x_N]^T$ の時、以下のように計算される。 

$$ y = f_(softmax) =[exp(x_1)/Σexp(x_i), exp(x_2)/Σexp(x_i), ... , exp(x_N)/Σexp(x_i)]^T ...[2.8] $$


アテンションの計算におけるソフトマックスへの入力は、クエリと各座標の特徴量の内積になる。  
よって、座標(x,y)のアテンションの値は以下のように計算される。

$$A(x,y) = f_(softmax) (q・F(x,y)) = exp(q・F(x,y))/Σ_uΣ_v exp(q・F(u,v)) .... [2.9] $$

A(x,y)が大きいほど特徴量とクエリの関連度が高く、重要な特徴であることを示す。

=code1.0=  
Attentionで特徴抽出を行う

```python

### 画像の特徴空間への射影 ###
'''
wはmatmul関数による行列積でＲＧＢの３チャネルを8チャネルに変換。
（例ではwの値は大きいアテンションの値が得られるようにあらかじめ計算されている）
'''

# 画像の読込
img = Image.opne("image_path")

#numpyを使うため画像をnumpy配列に変換
img = np.asarray(img, dtype = "float32")

# 画像を特徴空間に射影
w = np.array([[0.0065, -0.0045, -0.0018, 0.0075,
               0.0095. 0.0075, -0.0026, 0.0022],
               [-0.0065, 0.0081, 0.0097, -0.0070,
               -0.0086, -0.0107, 0.0062, -0.0050],
               [0.0024, -0.0018, 0.0002, 0.0023,
               0.0017, 0.0021, -0.0017, 0.0016]])

features = np.matmul(img, w)


### アテンションの計算 ###
'''
特徴量を抽出する。
これらをアテンション計算のクエリとし、matmul関数を使ってすべての特徴量と内積を計算。
'''

# アテンション計算用の特徴を画像から抽出
feature_white = features[50,50]
feature_pink = features[200,200]

# アテンションの計算
atten_white = np.matmul(features, feature_white)
atten_pink = np.matmul(features, feature_pink)

# ソフトマックスの計算
atten_white = np.exp(atten_white) / np.sum(np.exp(atten_white))
atten_pink = np.exp(atten_pink) / np.sum(np.exp(atten_pink))

### アテンションの表示 ###
'''
計算したアテンションの値をグレースケール画像で表示。アテンションの値は任意の値をとるので正規化を行う。
amax関数とamin関数を使って得られたアテンションの値の最大値と最小値を使って正規化を行うと、
最大のアテンションが1、最小のアテンションが0になる。
正規化後にNumPy配列をPIL画像に変換し表示する。
'''

# 表示用に最大・最小値で正規化
atten_white = Image.fromarray(atten_white - np.amin(atten_white)) / \
    (np.amax(atten_white) - np.amin(atten_white))

atten_pink = (atten_pink - np.amin(atten_pink)) / \
    (np.amax(atten_pink) - np.amin(atten_pink))

# NumPy配列をPIL画像に変換
img_atten_white = Image.fromarray(
    (atten_white * 255).astype('unit8'))
img_atten_pink = Image.fromarray(
    (atten_pink * 255).astype('unit8'))

print("白に対するアテンション")
display(img_atten_white)
print("ピンクに対するアテンション")
display(img_atten_pink)

```

※Attentionの重みやクエリの学習は別のファイル参照