# Perceptron
Implement a simple 2-features Perceptron with graphic plotting at each iteration.

## What is Perceptron?
感知器Perceptron (也稱為Perceptron Learning Algorithm簡稱PLA)
- 線性可分
 （2D：在平面上找一條線去完全切出這兩群；3D：的話就是可以在空間中找一個平面去切出兩群）
- 當w1*x1 + w2*x2 + w3*x3 +…. wn*xn >某一個定值(aka. threshold, -bias)，就會出發神經元發送信號出去
- Activation Function挑選像sign一樣導出離散分類結果的function

### Perceptron Algorithm
![perceptron algorithm](https://cdn-images-1.medium.com/max/1600/1*MofmXIxbv5AOIXHQp_hSOw.png)

Figure 1. Perceptron Algorithm (by Yeh James)

#### How to Learn?
平面的法向量為W, 法向量指向的輸入空間，其輸出值為+1，而與法向量反向的輸入空間，其輸出值為-1。

每次迭代:

1. Input為一筆資料的features，Perceptron將features分別跟Weights相乘再加bias，透過Activation二元分類情況的話將output經由Sign()將資料分成兩個結果[-1, 1]

2. 將所有Outputs與其相對之targets比較，找出其中一筆error Data出來update weights
用x 到原點的向量 X ,來修正直線的法向量 W

要怎麼修正呢？要看 x 的類別 y 來決定, 調整 W的方向

公式如下：
> W → W + y⋅X

用這個方法就可以改變直線的斜率,把分類錯誤的點 x ,歸到正確的一類。

3. 重新迭代直至沒有error為止


#### Pros & Cons
Perception優點：

- 最簡單的線性分類演算法，Perception演算法的原理可推廣至其他複雜的演算法。

Perception缺點：

- 一定要線性可分Perception演算法才會停下來（實務上我們沒辦法事先知道資料是否線性可分）
- Perception演算法的錯誤率不會逐步收斂
- Perception演算法只知道結果是A類還B類，但沒辦法知道是A, B類的機率是多少（Logistic regression可解決此問題）

#### Any Questions?
Q: What kind of problems can it fix?

A: Binary Classification (二元分類), a type of linear classifier


Q: Can it solve multiclass problems?

A: 有點麻煩，參考別人的例子：

Suppose we have data (𝑥1,𝑦1),…,(𝑥𝑘,𝑦𝑘) where 𝑥𝑖∈ℝ𝑛 are input vectors and 𝑦𝑖∈{red, blue, green} are the classifications.

We know how to build a classifier for binary outcomes, so we do this three times: group the outcomes together, {red, blue or green},{blue, red or green} and {green, blue or red}.



## Version
1.0

## Requirements
```
Python==3.5.2
numpy==1.15.4
pandas==0.18.1
```
## Coding Description
#### Activation Function: Sign
#### Error: count the numbers of error data


## Demo 

a binary classifier with Iris Data （經filter後有100筆資料）
以Iris 資料集來做資料線性可分的視覺化, 選出其中2種特徵(inputs) and2種花的種類(output)

#### Input datasets
```
# as a dataframe
from sklearn import datasets

iris_data = iris_data_preprocess()
iris_data.head()
```
```
   sepal length (cm)  petal length (cm)  target
0                5.1                1.4      -1
1                4.9                1.4      -1
2                4.7                1.3      -1
```

#### Build the Perceptron
```
myPerceptron = Perceptron(n_inputs=len(iris_data.columns)-1, save_fig=False)
```

#### Train
```
 myPerceptron.train(iris_data)
```

#### plot the final result
```
plot_data_and_line(iris_data, myPerceptron.weights, save_fig=False)
```



## References
Concepts:

- [第3.3講：線性分類-邏輯斯回歸(Logistic Regression) 介紹 by Yeh James](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-3%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E9%82%8F%E8%BC%AF%E6%96%AF%E5%9B%9E%E6%AD%B8-logistic-regression-%E4%BB%8B%E7%B4%B9-a1a5f47017e5)
- [機器學習--Perceptron Algorithm
 by Mark Chang](http://cpmarkchang.logdown.com/posts/189108-machine-learning-perceptron-algorithm)


Coding:

- [超簡單版perceptron learning algorithm實作及範例](http://terrence.logdown.com/posts/290508-python-simple-perceptron-learning-algorithm-implementations)

