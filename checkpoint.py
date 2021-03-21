import os
import pickle
import torch

class Checkpoint(object):

    def __init__(self, load_id=None, save_dict=None):
        # count all saved checkpoints
        self.__checkpoint_ids = self.__check_ids()

        # error handling
        assert [load_id, save_dict].count(None) == 1, "Please input either a load or save dict"
        if load_id:
            assert load_id in self.__checkpoint_ids, "The load-id you entered is invalid"
        elif save_dict:
            for k in [
                "generator", 
                "g_optimizer", 
                "discriminator", 
                "d_optimizer", 
                "step", 
                "iteration", 
                "samples", 
                "time", 
                "preview_noise",
                "loss_type",
                "dataset"]:
                assert k in save_dict.keys(), f"Missing dict_value \"{k}\""

        # give id to checkpoint
        self.__id = load_id if load_id is not None else self.__set_id()

        # model dictionary with all values
        self.__model_dict = self.__create_dict(save_dict)

    # repr method
    def __repr__(self):
        return f"Checkpoint({self._id})"

    # create list of all saved checkpoints
    def __check_ids(self):

        checkpoint_ids = []
        try:
            for cp in os.listdir("checkpoints"):
                cp = str(cp)
                if "checkpoint" in cp:
                    for el in ["checkpoint", ".pkl"]:
                        cp = cp.replace(el, "")
                    checkpoint_ids.append(int(cp))

        except OSError:
            os.mkdir("checkpoints")

        return checkpoint_ids

    # give new checkpoint a valid _id
    def __set_id(self):
        loop_range = 1 if not self.__checkpoint_ids else max(self.__checkpoint_ids) + 2
        for i in range(1, loop_range):
            if i not in self.__checkpoint_ids:
                new_id = i
                break
        return new_id

    # save model_dict
    def save(self, save_dict):
        with open(f"checkpoints/checkpoint{self.__id}.pkl" , 'wb') as f:
            pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

    # load model_dict
    def __load(self):
        with open(f"checkpoints/checkpoint{self.__id}.pkl", 'rb') as f:
            return pickle.load(f)

    # init dict var
    def __create_dict(self, model_dict=None):
        if model_dict is None:
            return self.__load()
        else:
            self.save(model_dict)
            return model_dict

    # give id of checkpoint
    def give_id(self):
        return self.__id

    # give model_dict
    def give_model_dict(self):
        return self.__model_dict
