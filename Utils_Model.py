import torch
import os


class Utils_Model:
    def __init__(self, model, model_name, model_epoch):
        self.model_save(model, model_name, model_epoch)

    def model_save(self, m, mn, me):
        if not os.path.exists("Results"):
            os.mkdir("Results")
        if not os.path.exists("Results/{}".format(mn)):
            os.mkdir("Results/{}".format(mn))
        torch.save(m, "Results/{}/{}_model.pt".format(mn, me))
        print("Model is saved to Results/{}/{}_model.pt".format(mn, me))

