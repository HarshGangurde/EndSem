#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix,ConfusionMatrixDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    classification_report,accuracy_score
)


# In[8]:


data=load_breast_cancer()
X=pd.DataFrame(data.data, columns=data.feature_names)
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[9]:


lr_model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
rf_model=RandomForestClassifier(random_state=42)
lr_model.fit(X_train,y_train)
rf_model.fit(X_train,y_train)

y_pred_lr=lr_model.predict(X_test)
y_pred_rf=rf_model.predict(X_test)
y_scores_lr=lr_model.predict_proba(X_test)[:, 1]
y_scores_rf=rf_model.predict_proba(X_test)[:, 1]


# In[10]:


cm_lr=confusion_matrix(y_test,y_pred_lr)
disp_lr=ConfusionMatrixDisplay(cm_lr,display_labels=data.target_names)
disp_lr.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("confusion_matrix_lr.png")
plt.close()

cm_rf=confusion_matrix(y_test,y_pred_rf)
disp_rf=ConfusionMatrixDisplay(cm_rf,display_labels=data.target_names)
disp_rf.plot()
plt.title("Confusion Matrix - Random Forest")
plt.savefig("confusion_matrix_rf.png")
plt.close()


# In[11]:


precision_lr, recall_lr, _=precision_recall_curve(y_test,y_scores_lr)
PrecisionRecallDisplay(precision_lr, recall_lr).plot()
plt.title("Precision-Recall Curve - Logistic Regression")
plt.savefig("pr_curve_lr.png")
plt.close()

precision_rf, recall_rf,_=precision_recall_curve(y_test,y_scores_rf)
PrecisionRecallDisplay(precision_rf,recall_rf).plot()
plt.title("Precision-Recall Curve - Random Forest")
plt.savefig("pr_curve_rf.png")
plt.close()


# In[12]:


print("Logistic Regression Report:\n", classification_report(y_test,y_pred_lr))
print("Random Forest Report:\n", classification_report(y_test,y_pred_rf))


# In[ ]:




