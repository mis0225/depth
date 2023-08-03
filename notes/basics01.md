
---
## 1.1 画像処理の基礎知識  

 **画素(Pixel)** と呼ばれる小さな格子によって構成されている。  
画素ごとの階調（色）を **画素値** と呼ぶ。
- pythonで画像を読み込んだ場合、この格子状の画素値の集合を２次元配列として保持し、処理することが一般的。
画像の左上を原点（0,0）とし、右方向とした方向をそれぞれx軸、y軸の正の方向とする。

- 色を表す軸を **チャネル(Channel)** という。
   - **グレースケール画像**は1チャネルの2次元配列でできており、配列の各要素は8bit（$2^8=256$）の値、つまり0~255の値となる。（0が黒、255が白）
   - **カラー画像**は3チャネルの２次元配列で、配列の各要素は8bitの3つの値となる。3つの値はそれぞれ赤、緑、青の色の強度を表していて、RGBの順に要素を構成する。（例：原点(0,0)の画素値＝(197, 200, 191)
   - カラー画像の各画素が表現できる色の数は$256^3=約1678万色$に及ぶ

=code1.0=  
PILを使って読み込んだ画像をNumPy配列に変換して、データを保持している配列の形を確認する 

画像処理ライブラリPIL（Python Image Library）：
画像を扱うのに必要な画像の読込や書き出しといった基本的な処理が実装

数値計算ライブラリNumPy：
画像を配列として扱う  



```python  
from PIL import Image
import numpy as np

#グレースケール画像の読み込みと表示
img_grey = Image.opne("image_path") #画像のパスを入れる
display(img_grey)

# 配列の形と画素値の確認
print("配列の形：{}".format(np.array(img_grey).shape))
print("画像の(0,0)における画素値：{}".format(img_grey.getpixel((0,0))))

## カラー画像も同様 ##
```

---

## 1.2 画像に平滑化フィルタをかける
**平滑化フィルタ（smoothing filter）** とは、  
画像の各座標における画素値とその周辺の画素値を考慮して更新する方法。  
ある画素値は、その近傍の画素値と近い値を持っている可能性が高いという局所性を利用してノイズ除去に用いられる。  
- 平均値フィルタ
- メディアんフィルタ
- **ガウシアンフィルタ**  


### **(1) カーネルの用意** ###

平均フィルタをかける際には、画素値を更新するためにどのように周辺の値を収集するかを決める **カーネル（Kernel）** を用意する。カーネルの各値は、周辺の画素値を収集する際の重みになっている。
  
ガウシアンフィルタでカーネルの値を求める式：  
カーネル中心を原点とする以下の式のような２次元ガウス分布によって決められる。 

$$G(x,y)= 1/√2πσ^2 exp(- (x^2+y^2)/ 2σ^2))  ...... [2.1]$$

- σはガウス分布における標準偏差
  - 任意の値を選ぶことができる  
  - σの値が大きくなればなるほど、カーネルの中心から離れた要素の重みが大きくなる

- カーネルについて
  - ガウス分布によって各値が決められたあと、総和が1になるように正規化される
  - カーネルは中心をもつように幅と高さは奇数を指定。3×3や5×5など正方形が用いられることが多い

### **(2) 畳み込み演算** ###

カーネルを用意したあと、カーネルと画像の間で **畳み込み演算（Convolution）** が行われる。これは、特徴抽出でも使われる重要な演算になる。  
画像上の座標(x,y)における畳み込み演算は、以下の式のように行われる。  
$$ I_g = K*I_O Σ[W/2],[u=-[W/2]]Σ[H/2], v=-[H/2] K(u+[W/2], c+[H/2])I_o(x+u,y+v) ....[2,2] $$  

「*」：畳み込み演算の演算子  
$I_O, I_g, K$ : フィルタ適用後の画像およびカーネルのその座標における値  
W, H :　それぞれのカーネルの幅と高さ

=code2.0=  
座標(x,y)の画素値を、その周辺の画素値をカーネルにより重みづけした値で更新。  
この操作を画像のすべての画素で行うことでフィルタ処理が完了する


```python

### ガウシアンカーネルを生成する関数 ###
'''
kernel_width: 生成するカーネルの幅
kernel_height: 生成するカーネルの高さ
sigma: カーネルの値を決めるガウス分布の標準偏差
'''

def generate_gaussian_kernel(kernel_width: int, kernel_height: int, sigma = float):
    # カーネルの大きさを奇数に限定
    assert kernel_width % 2 == 1 and kernel_height %2 == 1

    # カーネル用の変数を用意
    kernel = np.empty((kernel_height, kernel_width))
    
    for y in range(-(kernel_height // 2), kernel_height // 2 + 1):
        for x in range(-(kernel_width //2), kernel_width //2 + 1):
            # ガウス分布から値を抽出し、カーネルに代入
            h = -(x ** 2 + y ** 2) / (2* sigma ** 2)
            h = np.exp(h) / (2* np.pi * sigma ** 2)
            kernel[y + kernel_height // 2, x + kernel_width // 2] = h

    # カーネルの和が1になるように正規化
    kernel /= np.sum(kernel)

    return kernel  

### 畳み込み演算を行う関数 ###
'''
画像上の座標(x,y)を中心としてカーネルの高さと幅の分だけカーネルを走査し、画素値とカーネルの値の積をvalueに蓄積。
カーネルが画像の端にある時、抽出する画像上の座標を画像端の値で丸め込み、画像端の値を抽出する。

img : 適用する画像
kernel : 平滑化のカーネル,[カーネル高さ,カーネル幅]
'''

def convolution(img: Image.Image, kernel: np.ndarray, x: int, y: int):
    # 画像サイズとカーネルサイズの取得
    width, height = img.size
    kernel_height, kernel_width = kernel.shape[:2]
    value = 0
    for y_kernel in range(-(kernel_height // 2),kernel_height // 2 +1):
        for x_kernel in range(-(kernel_width // 2), kernel_width // 2+ 1):
            # カーネルが画像からはみ出る場合、端の座標を取得
            x_img = max(min(x + x_kernel, width - 1), 0)
            y_img = max(min(y + y_kernel, height -1), 0)
            h = kernel[y_kernel + kernel_height // 2, x_kernel + kernel_width // 2]
            value += h * img.getpixel((x_img, y_img))
    return value

### 画像にカーネルを適用する関数 ###
'''
Imageモジュールのnew関数を使って入力画像と同じ大きさの画像img_filteredを用意。
その後forループを使って画像上の各座標で畳み込み演算を行い、得られた値をputpixel関数を使ってimg_filteredに代入。
数値計算ライブラリを活用して効率的に畳み込みを計算することも可能
'''

def apply_filteer(img: Image.Image, kernel: np.ndarray):
    # 画像サイズとカーネルサイズの取得
    width, height = img.size

    # フィルタ適用後の画像を保持する変数を用意
    img_filtered = Image.new(mode='L', size = (width, height))

    # フィルタ適用後の各画素値の計算
    for y in range(height):
        for x in range(width):
            filtered_value = convolution(img, kernel, x, y)
            img_filtered.putpixel((x,y), int(filtered_value))
    
    return img_filtered

## ガウシアンカーネルの生成 ##

kernel = generate_gaussian_kernel(
    kernel_width=5, kernel_height=5, sigma-1.3)

## ガウシアンフィルタの適用 ##

# 画像の読込
img = Image.open("image_path")

# ガウシアンフィルタの適用
img_filtered = apply_filter(img, kernel)

# 元画像とフィルタ適用後の画像の表示
print("元のノイズあり画像")
display(img)
print("フィルタ適用後の画像")
display(img_filtered)

```
