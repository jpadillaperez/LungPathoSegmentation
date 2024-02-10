import numpy as np
import torch



class RemovePleuralEffusion(object):

    def __call__(self, sample):
            split_images = torch.split(sample, 1, dim=0)
            results = []
            for i in range(len(split_images)):
                label_ref = split_images[i]
                label_ref = self.remove_pleural_effusion(label_ref)
                label_ref = torch.squeeze(label_ref, 0)
                results.append(label_ref)
            result = np.stack(results, axis=0)
            return torch.tensor(result)



    def remove_pleural_effusion(self, label_ref):
        label_ref_clone = label_ref.clone()
        label_ref_clone[label_ref_clone > 3] = 0.
        label_ref_new = label_ref_clone
        return torch.tensor(label_ref_new)

