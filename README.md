# 🧠 Machine Learning Cheat Sheets — Complete Guide

Welcome to the **Machine Learning Cheat Sheets** repository!  
This repo provides concise, professional cheat sheets covering **core ML concepts, formulas, and Python examples** — ideal for students, developers, and professionals.

---

## 📘 Table of Contents

| No. | Cheat Sheet | Description |
|----|--------------|-------------|
| 01 | **ML Basics** | Overview of ML types, workflows, and key terms |
| 02 | **Data Preprocessing** | Techniques for cleaning and preparing data |
| 03 | **Regression Models** | Linear, Ridge, Lasso, and evaluation metrics |
| 04 | **Classification Models** | Logistic, SVM, KNN, Naive Bayes, etc. |
| 05 | **Clustering Models** | K-Means, DBSCAN, Hierarchical clustering |
| 06 | **Dimensionality Reduction** | PCA, LDA, and visualization methods |
| 07 | **Model Evaluation** | Confusion Matrix, ROC-AUC, F1-score |
| 08 | **Feature Engineering** | Selection, extraction, and encoding techniques |
| 09 | **Ensemble Methods** | Random Forest, XGBoost, LightGBM |
| 10 | **Neural Networks** | Activation functions and backpropagation |
| 11 | **Deep Learning** | CNNs, RNNs, LSTMs, and optimizers |
| 12 | **Computer Vision** | CNN architectures, object detection |
| 13 | **NLP Basics** | Text preprocessing, embeddings, sentiment |
| 14 | **Time Series Analysis** | ARIMA, Prophet, forecasting techniques |
| 15 | **ML Libraries** | Numpy, Pandas, Scikit-learn, TensorFlow, PyTorch |

---

## 🧮 Machine Learning Formulas & Key Concepts

---

### 🟩 01. Machine Learning Basics
**Formulas**
- **Linear Model:**  
  $$y = f(x) + \epsilon$$  
- **Hypothesis Function:**  
  $$\hat{y} = \theta_0 + \theta_1x_1 + \dots + \theta_nx_n$$  
- **Cost Function (MSE):**  
  $$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y_i} - y_i)^2$$  
- **Gradient Descent:**  
  $$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$  

---

### 🟩 02. Data Preprocessing
**Formulas**
- **Standardization:**  
  $$z = \frac{x - \mu}{\sigma}$$  
- **Min-Max Normalization:**  
  $$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$  
- **Z-Score:**  
  $$z_i = \frac{x_i - \bar{x}}{s}$$  
- **IQR Range:**  
  $$IQR = Q3 - Q1$$  

---

### 🟩 03. Regression Models
**Formulas**
- **Simple Linear Regression:**  
  $$\hat{y} = \beta_0 + \beta_1x$$  
- **Multiple Linear Regression:**  
  $$\hat{y} = \beta_0 + \sum_{i=1}^n \beta_i x_i$$  
- **Ridge (L2) Regression:**  
  $$J(\beta) = \frac{1}{2m}\sum(\hat{y}-y)^2 + \lambda \sum \beta_j^2$$  
- **Lasso (L1) Regression:**  
  $$J(\beta) = \frac{1}{2m}\sum(\hat{y}-y)^2 + \lambda \sum |\beta_j|$$  

---

### 🟩 04. Classification Models
**Formulas**
- **Logistic Regression (Sigmoid):**  
  $$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x)}}$$  
- **Naive Bayes:**  
  $$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$  
- **KNN Distance (Euclidean):**  
  $$d(x,y) = \sqrt{\sum (x_i - y_i)^2}$$  
- **SVM Optimization:**  
  $$\min \frac{1}{2}\|w\|^2 \ \text{s.t. } y_i(w·x_i + b) ≥ 1$$  

---

### 🟩 05. Clustering Models
**Formulas**
- **K-Means Objective:**  
  $$\min \sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \mu_k\|^2$$  
- **Centroid Update:**  
  $$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$  
- **Silhouette Score:**  
  $$s = \frac{b - a}{\max(a,b)}$$  

---

### 🟩 06. Dimensionality Reduction
**Formulas**
- **Covariance Matrix:**  
  $$C = \frac{1}{n-1}(X - \bar{X})^T(X - \bar{X})$$  
- **PCA Eigen Equation:**  
  $$Cw = \lambda w$$  
- **Projection of Data:**  
  $$Y = XW$$  

---

### 🟩 07. Model Evaluation
**Formulas**
- **Accuracy:**  
  $$\frac{TP + TN}{TP + TN + FP + FN}$$  
- **Precision:**  
  $$\frac{TP}{TP + FP}$$  
- **Recall:**  
  $$\frac{TP}{TP + FN}$$  
- **F1 Score:**  
  $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$  
- **AUC:**  
  $$AUC = \int_0^1 TPR(FPR) \, d(FPR)$$  

---

### 🟩 08. Feature Engineering
**Formulas**
- **Polynomial Expansion:**  
  $$x_{new} = [x_1, x_2, x_1^2, x_1x_2, x_2^2, ...]$$  
- **Correlation Coefficient:**  
  $$r = \frac{Cov(X,Y)}{\sigma_X\sigma_Y}$$  
- **Entropy:**  
  $$H(Y) = -\sum p(y)\log_2 p(y)$$  
- **Information Gain:**  
  $$IG = H(Y) - H(Y|X)$$  

---

### 🟩 09. Ensemble Methods
**Formulas**
- **Bagging Average:**  
  $$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(x)$$  
- **Boosting Update:**  
  $$F_m(x) = F_{m-1}(x) + \alpha_m h_m(x)$$  
- **Weighted Error (AdaBoost):**  
  $$\epsilon_m = \frac{\sum w_i I(y_i \neq h_m(x_i))}{\sum w_i}$$  
- **Weight Update:**  
  $$w_i := w_i e^{\alpha_m I(y_i \neq h_m(x_i))}$$  

---

### 🟩 10. Neural Networks
**Formulas**
- **Neuron Output:**  
  $$z = w^Tx + b$$  
- **Activation Functions:**  
  - Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$  
  - ReLU: $f(z) = \max(0, z)$  
  - Tanh: $f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$  
- **Cost Function:**  
  $$J = \frac{1}{m} \sum (y - \hat{y})^2$$  
- **Gradient Descent:**  
  $$w := w - \alpha \frac{\partial J}{\partial w}$$  

---

### 🟩 11. Deep Learning
**Formulas**
- **Convolution Operation:**  
  $$(I * K)(x, y) = \sum_m \sum_n I(x+m, y+n)K(m,n)$$  
- **LSTM Gates:**  
  $$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$$  
  $$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$$  
- **Cell State Update:**  
  $$C_t = f_t * C_{t-1} + i_t * \tilde{C_t}$$  
- **Adam Optimizer:**  
  $$\theta_t = \theta_{t-1} - \eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}$$  

---

### 🟩 12. Computer Vision
**Formulas**
- **Image Normalization:**  
  $$I_{norm} = \frac{I - \mu}{\sigma}$$  
- **CNN Filter Output:**  
  $$O = I * K + b$$  
- **Softmax Function:**  
  $$P(y_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$  
- **Cross-Entropy Loss:**  
  $$L = -\sum y_i \log(\hat{y_i})$$  

---

### 🟩 13. NLP Basics
**Formulas**
- **TF-IDF Weight:**  
  $$w_{t,d} = tf_{t,d} \times \log(\frac{N}{df_t})$$  
- **Cosine Similarity:**  
  $$\cos(\theta) = \frac{A \cdot B}{\|A\|\|B\|}$$  
- **Word2Vec Skip-Gram:**  
  $$P(w_O|w_I) = \frac{e^{v_{w_O}^T v_{w_I}}}{\sum_{w=1}^W e^{v_w^T v_{w_I}}}$$  
- **Loss Function:**  
  $$L = -\sum y_i \log(\hat{y_i})$$  

---

### 🟩 14. Time Series Analysis
**Formulas**
- **AR Model:**  
  $$y_t = c + \phi_1 y_{t-1} + \epsilon_t$$  
- **MA Model:**  
  $$y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1}$$  
- **ARIMA (p,d,q):**  
  $$\phi(B)(1 - B)^d y_t = \theta(B)\epsilon_t$$  
- **MAPE:**  
  $$MAPE = \frac{100}{n}\sum \left| \frac{y_t - \hat{y_t}}{y_t} \right|$$  
- **RMSE:**  
  $$RMSE = \sqrt{\frac{1}{n}\sum (y_t - \hat{y_t})^2}$$  

---

### 🟩 15. ML Libraries Quick Formulas
**Common Operations**
- **NumPy Matrix Multiplication:**  
  $$A \cdot B = \sum a_i b_i$$  
- **Scikit-Learn Pipeline:**  
  $$Pipeline = [(‘scaler’, StandardScaler()), (‘model’, LogisticRegression())]$$  
- **TensorFlow Dense Layer:**  
  $$z = Wx + b$$  
- **PyTorch Backprop:**  
```
  loss.backward()
  optimizer.step()
 ```
---
## 👨‍💻 Author

---
### Adarsh Lilhare 

🎓 B.Tech in Artificial Intelligence & Data Science

💼 AI & Data Science Student | 💻 Developer | 🌍 Open Source Contributor

📧 [Email](adarshlilhare@example.com)

🐙 [GitHub](https://github.com/AdarshVL) 

🌐 [Portfolio](https://adarshlilhare.dev)

🔗 [LinkedIn](https://www.linkedin.com/in/adarsh-lilhare-b98a91290/)

---
