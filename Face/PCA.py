import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

if __name__ == '__main__':

    # 3차원 데이터 60개를 랜덤으로 생성
    n_data = 60
    X = np.empty((n_data, 3)) # 행이 3개
    noise = 0.1

    # np.random.rand 0~1 사이 uniform ==> 0~1사이 값이 일정하게 나온다. (uniform) : 균일분포(난수 matrix array 생성)
    # np.random.randn norm 분포 ==> 평균과 표준편차를 가지고 있는 정규분포가 나온다.(norm) : 표준 정규분포(난수 matrix array 생성)
    #cosine 값을 넣어준다.
    X[:, 0] = np.cos(np.random.rand(n_data)) // 2 + noise * np.random.randn(n_data) / 2
    X[:, 1] = np.sin(np.random.rand(n_data)) * 0.7 + noise * np.random.randn(n_data) / 2
    X[:, 2] = X[:,0] * 0.1 + X[:, 1] * 0.3 + noise * np.random.randn(n_data) / 2
    print(X.shape)

    df = pd.DataFrame(X, columns=["X1", "X2", "X3"])
    # print(df.head(10))

    pca = PCA(n_components=0.90) # 90퍼센트까지! | integer로 주면 차원 수로 출력된다.
    X_reduced = pca.fit_transform(X) # X 차원을 줄여서 fit해준다.

    print(X_reduced.shape) # 차원을 줄여서 변환까지 한 것이 X_reduced이다.

    print("Eigen Value:", pca.explained_variance_)
    print("exp var ratio:", pca.explained_variance_ratio_)
    print("선택한 차원수:", pca.n_components_)

    # 데이터의 컬럼의 수가 엄청 많을 때(데이터 양이 너무 많아서 학습이 너무 오래 걸린다.)
    # 모델 자체를 PCA로 만들었을 때는 데이터를 가져올 때도 PCA 해서 가져와서 모델에 집어넣어야 한다.



