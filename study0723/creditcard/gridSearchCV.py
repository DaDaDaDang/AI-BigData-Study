from creditcard.smoteExample import *

def gridSearchCV(model, x_smote, y_smote) :
    parameters = {
        'C': np.linspace(1, 10, 10),
        'penalty' : ['l1', 'l2']
    }
    gscv = GridSearchCV(model, parameters, cv=5 , verbose=5, n_jobs = 3)

    print("<<clt - fit>>")
    gscv.fit(x_smote, y_smote.ravel())

    print("<<Best params>>", gscv.best_params_, gscv.best_estimator_, gscv.best_score_)
