class ModelPredictionFactory:
    def __init__(self):
        pass

    def get_predictions(self, model, new_data):
        # TODO: error checks
        preds = model.predict(new_data)
        return preds

    def save_predictions(self, preds, file_name):
        # TODO: error checks
        save(preds)

    def get_and_save_predictions(self, model, new_data, pred_file_name):
        preds = self.get_predictions(model=model, new_data=new_data)
        self.save_predictions(preds=preds, file_name=pred_file_name)
        return preds