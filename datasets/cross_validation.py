import numpy as np

# data split for test set
class TestSplit:

    def get_idxs_split(self):
        dict = {}
        dict['train'] = np.array([])
        dict['val'] = np.array([])
        dict['test'] = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        return dict

# data split for Training and Validation
class TrainValSplit:

    def get_idxs_split(self):
        dict = {}
        dict['train'] = np.array(
            ['W1_Unknown24', 'W1_MdpjxIs6qiQ', 'W1_Unknown14', 'W1_Rir6kafonJw', 'W1_i6oVO2TKMlo', 'W1_5cTDobtjJMY',
             'W1_Unknown29', 'W1_Unknown27', 'W1_xD70fSZbMxg', 'W1_Unknown19', 'W1_17IZtmjT2dc',
             'W1_ddDGQK1d0Eg', 'W1_Unknown12', 'W1__fd7TI97ccc', 'W1_Unknown28', 'W1_Unknown36', 'W1_Unknown2',
             'W1_nbAeU14Q-Ic', 'W1_cgtif4FXuKo', 'W1_Wnlafu596HU', 'W1_ErMtk-DlIAwß', 'W1_CtU-Kjdkyk8',
             'W1__5DrME2a0BI', 'W1_Unknown16', 'W1_Unknown35', 'W1_wHxmGo7W6cs', 'W1_Unknown50', 'W1_4M54cTpoWvA',
             'W1_KYol8VCMuTQ', 'W1_Unknown48']
        )

        dict['val'] = np.array(
            ['W1_Sh_WCs_arH4', 'W1_Unknown26', 'W1_1mVhVpe9H2w', 'W1_U-TvAUGnlf8', 'W1_bl20K5m928k', 'W1_Unknown42',
             'W1_Unknown15']
        )
        dict['test'] = np.array([])
        return dict
