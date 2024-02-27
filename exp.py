import os
from model_factory import GLMFactory

class Exp:

    def __init__(self, exp_dir, model, model_params, data, data_params):
        self.exp_dir = exp_dir
        self.model = model
        self.model_params = model_params
        self.data = data
        self.data_params = data_params
        self.setup()

    def setup(self):
        self._setup_dirs()
        self._setup_data()
        self._setup_model()

    def _setup_dirs(self):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

    def _setup_data(self):
        data_type = self.data_params.pop("data_type")
        if data_type == "nephqa":
            pass
        else:
            raise ValueError("Data type {} not supported".format(data_type))
        self.data = pass

    def _setup_model(self):
        model_type = self.model_params.pop("model_type")
        if model_type == "greaselm":
            model_factory = GLMFactory()
        else:
            raise ValueError("Model type {} not supported".format(model_type))
        self.model = model_factory.create(self.model_params)

    def run(self):
        self.model.train()
        self.model.infer_test()