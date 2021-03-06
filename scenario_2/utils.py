import datetime 

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
        self.d_train = lgb.Dataset(x_train, label=y_train)
        self.d_test = lgb.Dataset(x_test, label=y_test)
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
            return {'status': STATUS_FAIL, 'exception': str(e)}

        return result, trials

    def lgb_clf(self, para):
        """
        Create and train a model
        :param para: model and fitting parameters
        :return:
        """
        with mlflow.start_run(experiment_id=self.mlflow_exp_id):
            clf = None
            # Log parameters to MLFlow
            for k, v in para['params_space'].items():
                mlflow.log_param(k, v)
            train_res = self.__train_clf(clf, para)

        return train_res

    def __train_clf(self, clf, para):
        """
        Train and evaluate the model
        :param clf: classifier
        :param para: parameters dictionary
        :return:
        """
        params = para['fit_params']
        params.update(para['params_space'])
        clf = lgb.train(params, self.d_train,
                        valid_sets=[self.d_train, self.d_test],
                        valid_names=['train', 'valid'])
        # Log Score
        train_score = para['scoring_func'](self.y_train, clf.predict(self.x_train).argmax(axis=1))
        mlflow.log_metric('train_score', train_score)
        test_score = para['scoring_func'](self.y_test, clf.predict(self.x_test).argmax(axis=1))
        mlflow.log_metric('val_score', test_score)

        loss = clf.best_score['valid']['multi_logloss']

        return {'loss': loss, 'status': STATUS_OK}
