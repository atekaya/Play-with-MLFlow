import datetime 

from sklearn.metrics import log_loss

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL

import lightgbm as lgb

import mlflow


class HPOpt(object):
    """
    HyperOpt object to create the model, train it, and evaluate it
    Credit: https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e
    """

    def __init__(self, x_train, x_test, y_train, y_test, experiment_name):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        # Create MLFlow experiment
        self.mlflow_exp_id = mlflow.create_experiment(
            name=experiment_name+'_'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        )

    def process(self, space, trials, algo, max_evals):
        """
        Perform hyper-parameter tuning using HyperOpt
        :param space: Parameter space
        :param trials:
        :param algo:
        :param max_evals: Number of trials
        :return:
        """
        try:
            
                result = fmin(fn=self.lgb_clf, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def lgb_clf(self, para):
        """
        Create and train a model
        :param para: model and fitting parameters
        :return:
        """
        with mlflow.start_run(experiment_id=self.mlflow_exp_id):
            clf = lgb.LGBMClassifier(**para['params_space'])
            # Log parameters to MLFlow
            for k, v in para['params_space'].items():
                mlflow.log_param(k, v)
            train_res = self.train_clf(clf, para)

        return train_res

    def train_clf(self, clf, para):
        """
        Train and evaluate the model
        :param clf: classifier
        :param para: parameters dictionary
        :return:
        """
        clf.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])

        pred_proba = clf.predict_proba(self.x_test)
        loss = log_loss(self.y_test, pred_proba)

        pred_class = clf.predict(self.x_test)
        score = para['scoring_func'](self.y_test, pred_class)

        # Log metrics: loss and score
        mlflow.log_metric('val_loss', loss)
        mlflow.log_metric('val_score', score)

        return {'loss': loss, 'status': STATUS_OK}
