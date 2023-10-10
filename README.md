# coding_festival


1.  숫자 선택


```
# 입력 받기
N = int(input())  # 숫자의 개수 N
numbers = list(map(int, input().split()))  # N개의 숫자

# 초기 합계를 0으로 설정
total_sum = 0

# 숫자들을 반복하면서 짝수이면서 짝수 번째 위치에 있는 숫자를 합산
for i in range(N):
    if numbers[i] % 2 == 0 and (i + 1) % 2 == 0:
        total_sum += numbers[i]
        
# 결과 출력
print(total_sum)
```





2. 레이저
```
N, h = map(int, input().split())

# 판자 정보를 입력받는다.
boards = []
for _ in range(N):
    x, s, k = map(int, input().split())
    boards.append((x, x+1, s, k))

  
# 판자의 x 좌표를 기준으로 오름차순 정렬
boards.sort(key=lambda x: x[0])

hole_count = 0  # 구멍이 뚫리는 판자의 수
stop_x = -1  # 레이저가 멈추는 지점

  

# 각 판자를 순회하며 레이저의 교차 여부를 판단
for board in boards:
    # 레이저가 판자와 교차하는 경우
    if h < board[2]:
        if board[3] == 1:
            hole_count += 1
        elif board[3] == 3:
            stop_x = board[0]
            break
        elif board[3] == 4:
            hole_count += 1
            
# 결과 출력

print(stop_x, hole_count)
```





3. 소설 속 단어의 중복

```
# 첫 번째 줄에 문자열을 입력받습니다.
word_to_search = input().strip()


# 파일을 열어 내용을 불러옵니다.
with open("Harry_Potter.txt", "r") as file:
    content = file.read()


# 파일의 내용을 공백을 기준으로 분리하여 리스트로 변환합니다.
words = content.split()


# 입력받은 문자열과 동일한 단어가 반복되는 횟수를 구합니다.
count = words.count(word_to_search)

# 결과를 출력합니다.
print(count)
```







4. 자동차 사고 조건 분석

```
import pandas as pd

# csv 파일 불러오기
df = pd.read_csv('car_crashes.csv')

# 조건으로 비교할 컬럼명과 비교 조건 숫자를 입력받기
column_name = input()
condition_number = float(input())

# 조건에 따라 데이터 필터링
filtered_data = df[df[column_name] > condition_number]


# 상위 10줄 출력
print(filtered_data.head(10))
```







5. 식당 방문 손님 일치 조건
```
import pandas as pd

  

# tips.csv 파일 불러오기

df = pd.read_csv("tips.csv")

  

# 조건으로 사용할 컬럼명 입력받기

column_name = input()

  

# 해당 컬럼이 문자 형식인지 확인

if df[column_name].dtype == "object":

    # 내용과 비교할 문자열 입력받기

    value = input()

    # 조건에 맞는 데이터 출력

    matched_rows = df[df[column_name] == value]

    print(matched_rows.head(20))

else:

    print(f"{column_name} is not a string column.")
```









6. 전력 데이터 탐색


```
import pandas as pd

  

# 2019년의 데이터만 추출하는 함수

def get_2019(df: pd.DataFrame) -> pd.DataFrame:

    return df[df["Year"] == 2019]

  

# 시간별 평균 소비량을 구하는 함수

def hour_consum_mean(df: pd.DataFrame) -> pd.Series:

    return df.groupby("Hour")["Consumption"].mean()

  

# 연도가 2020이고 전력 소비량이 6000보다 큰 데이터를 추출하는 함수

def big_consum_2020(df: pd.DataFrame) -> pd.DataFrame:

    return df[(df["Year"] == 2020) & (df["Consumption"] > 6000)]
```






7. 이미지 클러스터링

```
from PIL import Image

import numpy as np

from sklearn.cluster import KMeans

  

def load_img():

    return Image.open("img.jpg")

  

def reshape_img(img: Image) -> np.ndarray:

    # 1. 이미지를 np.ndarray로 변환

    np_img = np.array(img)

    # 2. 이미지의 shape 확인 후 한 줄로 만들기

    np_img = np_img.reshape(-1, 3)

    return np_img

  

def k_means(np_img: np.ndarray, n_clusters: int) -> KMeans:

    # 1. KMeans 객체 생성

    kmeans = KMeans(n_clusters=n_clusters, random_state=10)

    # 2. kmeans를 사용해 클러스터링

    kmeans.fit(np_img)

    return kmeans

  

def cvt_color(np_img: np.ndarray, kmeans: KMeans) -> np.ndarray:

    # kmeans를 사용하여 이미지의 색상을 변경

    centers = kmeans.cluster_centers_

    labels = kmeans.labels_

    # 각 픽셀의 색상을 해당 픽셀의 속하는 클러스터의 중앙값으로 변경

    for i in range(len(np_img)):

        np_img[i] = centers[labels[i]]

    return np_img

  

def main():

    img = load_img()

    np_img = reshape_img(img)

    kmeans = k_means(np_img, 5)

    np_img = cvt_color(np_img, kmeans)

  

    # 이미지 변환된 내용을 확인하려면 아래 코드를 추가하여 변환된 이미지를 저장 및 확인

    # out_img = Image.fromarray(np_img.reshape(img.size[1], img.size[0], 3).astype('uint8'))

    # out_img.save('output_img.jpg')

    # out_img.show()

  

if __name__ == "__main__":

    main()
```

    





8. SVM 학습하기


```
from typing import List

import numpy as np

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, recall_score

  
  

def load_data() -> tuple:

    """데이터를 불러오는 함수"""

  

    # 데이터 불러오기

    iris = datasets.load_iris()

    X = iris.data

    y = iris.target

  

    # 데이터 나누기

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  

    return X_train, X_test, y_train, y_test

  
  

def data_scaling(X_train: np.ndarray, X_test: np.ndarray) -> tuple:

    """데이터 스케일링 함수"""

  

    # 데이터 스케일링

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

  

    return X_train_scaled, X_test_scaled

  
  

def train_svm(x_train: np.ndarray, y_train: np.ndarray) -> SVC:

    """SVM 모델을 학습하는 함수"""

  

    # SVM 모델 생성 및 학습z

    svm_model = SVC(kernel='linear', C=1.0)

    svm_model.fit(x_train, y_train)

  
  

    return svm_model

  
  

def validate_svm(svm_model: SVC, x_test: np.ndarray, y_test: np.ndarray) -> List[float]:

    """학습된 SVM 모델을 검증하는 함수"""

  

    # 테스트 데이터에 대한 예측

    y_pred = svm_model.predict(x_test)

  

    # 정확도 계산

    accuracy = accuracy_score(y_test, y_pred)

  

    # 정답 클래스가 1인 경우에 대한 재현율 계산

    binary_y_test = np.where(y_test == 1, 1, 0)

    binary_y_pred = np.where(y_pred == 1, 1, 0)

    recall = recall_score(binary_y_test, binary_y_pred)

  

    return accuracy, recall

  
  

def main():

    """SVM 모델 학습 및 검증 메인 함수"""

  

    # 데이터 불러오기

    X_train, X_test, y_train, y_test = load_data()

  

    # 데이터 스케일링

    X_train_scaled, X_test_scaled = data_scaling(X_train, X_test)

  

    # SVM 모델 학습

    svm_model = train_svm(X_train_scaled, y_train)

  

    # SVM 모델 검증

    accuracy, recall = validate_svm(svm_model, X_test_scaled, y_test)

  

    print(f"Accuracy: {accuracy:.2f}")

    print(f"Recall: {recall:.2f}")

  
  

if __name__ == "__main__":

    main()
```








9. 랜덤포레스트를 이용하여 다이아몬드 가격 예측하기

```
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_percentage_error

  

# pd.set_option 설정

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

  

def load_csv(path):

    return pd.read_csv(path)

  

def divide_data(df):

    # 독립변수와 종속변수 나누기

    X = df.drop("price", axis=1)

    y = df["price"]

    # 학습용 데이터와 검증용 데이터로 나누기 (비율 8:2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

  

def model_train(X_train, y_train):

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    return model

  

if __name__ == "__main__":

    path = "train.csv"

    df = load_csv(path)

    # 범주형 변수를 원핫인코딩으로 변환

    df = pd.get_dummies(df, columns=["cut", "color", "clarity"], drop_first=True)

    X_train, X_test, y_train, y_test = divide_data(df)

    model = model_train(X_train, y_train)

    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("MAPE:", mape)
```







10. 손글씨 분류하기

```
import sys

import warnings

  

import numpy as np

from sklearn.datasets import load_digits

from sklearn.neural_network import MLPClassifier

  

warnings.filterwarnings(action="ignore")

np.random.seed(100)

  
  

def load_data(X, y):

    """1. 손글씨 데이터를 X, y로 읽어온 후

    학습 데이터, 테스트 데이터로 나눕니다.

  

    Step01. 학습 데이터는 앞의 1600개를 사용하고,

            테스트 데이터는 학습 데이터를 제외한 나머지를 사용합니다.

            X, y 데이터의 타입은 NumPy array라는 것을 참고하세요.

    """

  

    X_train = X[:1600]

    Y_train = y[:1600]

  

    X_test = X[1600:]

    Y_test = y[1600:]

  

    return X_train, Y_train, X_test, Y_test

  
  

def train_MLP_classifier(X, y, hidden_layers=(50,50)):

    """2. MLPClassifier를 정의하고 hidden_layer_sizes를

    조정해 hidden layer의 크기 및 레이어의 개수를

    바꿔본 후, 학습을 시킵니다.

    """

  

    clf = MLPClassifier(hidden_layer_sizes=hidden_layers,

    solver='adam',

    beta_1=0.999999)

  

    clf.fit(X, y)

    return clf

  
  

def report_clf_stats(clf, X, y):

    """3. 정확도를 출력하는 함수를 완성합니다.

    이전 실습에서 작성한 "score"를 그대로

    사용할 수 있습니다.

    """

  

    hit = 0

    miss = 0

  

    for x, y_ in zip(X, y):

        if clf.predict([x])[0] == y_:

            hit += 1

        else:

            miss += 1

  

    score = hit/(hit+miss)

  

    print(f"Accuracy: {score:.1f} ({hit} hit / {miss} miss)")

  

    return score

  
  

def main():

    """4. main 함수를 완성합니다.

  

    Step01. 훈련용 데이터와 테스트용 데이터를

            앞에서 완성한 함수를 이용해 불러옵니다.

  

    Step02. 앞에서 학습시킨 다층 퍼셉트론 분류

            모델을 "clf"로 정의합니다.

  

    Step03. 앞에서 완성한 정확도 출력 함수를

            "score"로 정의합니다.

    """

  

    digits = load_digits()

  

    X = digits.data

    y = digits.target

  

    X_train, Y_train, X_test, Y_test = load_data(X, y)

  

    clf = train_MLP_classifier(X_train, Y_train, hidden_layers=(50,50))

  

    score = report_clf_stats(clf, X_test, Y_test)

  

    return 0

  
  

if __name__ == "__main__":

    sys.exit(main())
```


