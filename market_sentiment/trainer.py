# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier,VotingClassifier
# from sklearn.model_selection import train_test_split,cross_validate
# from sklearn.ensemble import  VotingClassifier
# from market_sentiment import save_model


# #mean_accuracy = cross_validate(voting_classifier_soft,X,y,cv=7,scoring=['f1','accuracy'])['test_accuracy'].mean()
# #voting_classifier_soft.fit(X_train,y_train)


# class Trainer(object):
#     def __init__(self, X, y):
#         """
#             X: pandas DataFrame
#             y: pandas Series
#         """
#         self.voting_classifier_soft = None
#         self.X = X
#         self.y = y
#         # for MLFlow
#     def set_model(self):
#         self.voting_classifier_soft = VotingClassifier(estimators = [('C-Support Vector Classification',SVC(probability=True)),
#                                                        ('Random Forest Classifier',RandomForestClassifier())],
#                                          voting='soft')
#     def run(self):
#         self.set_model()
#         self.voting_classifier_soft.fit(self.X, self.y)
#     def save_model_locally(self):
#         save_model(self.voting_classifier_soft)
# if __name__ == "__main__":
#     pass
