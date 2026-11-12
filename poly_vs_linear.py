import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

data=sns.load_dataset("mpg")
data.dropna(inplace=True)

X=data[['displacement']]
y=data['mpg']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
linear_model=LinearRegression()
linear_model.fit(X_train,y_train)
y_pred_linear=linear_model.predict(X_test)
print("LinearRegression")
print("---------")
print("MSE :",mean_squared_error(y_test,y_pred_linear))
print("R2:",r2_score(y_test,y_pred_linear))

degrees=[2,3,4]
colors=['red','green','orange']
plt.figure(figsize=(10,6))
plt.scatter(X['displacement'],y,
            color='blue',
            alpha=0.5,
            label='Actual Data')
X_plot=pd.DataFrame({
    'displacement':np.linspace(X['displacement'].min(),X['displacement'].max(),300)
})
for degree,color in zip(degrees,colors):
    poly=PolynomialFeatures(degree=degree)
    X_train_poly=poly.fit_transform(X_train)
    X_test_poly=poly.transform(X_test)
    model=LinearRegression()
    model.fit(X_train_poly,y_train)
    y_pred=model.predict(X_test_poly)
    print(f"\nPolynomial Degree{degree}")
    print("-----")
    print("MSE :",r2_score(y_test,y_pred))
    X_plot_poly=poly.transform(X_plot)
    y_plot=model.predict(X_plot_poly)
    plt.plot(X_plot['displacement'],y_plot,color=color,linewidth=2,label=f'Degree{degree}')
    X_line=pd.DataFrame({'displacement':np.linspace(X['displacement'].min(),X['displacement'].max(),300)})
    y_line=linear_model.predict(X_line)
    plt.plot(X_line['displacement'],y_line,color='black',linewidth=2,label='Linear Regression')
    plt.xlabel("Engine Displacement")
    plt.ylabel("Miles Per Gallon(MPG)")
    plt.title("Polynomial Regression vs Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.show()
