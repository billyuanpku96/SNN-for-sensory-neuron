import numpy as np
import pandas as pd
import spikingjelly.clock_driven.examples.lif_fc_mnist as lif_fc_mnist

train_loss = np.load('try22/train_loss.npy')
train_accs= np.load('try22/train_accs.npy')
test_accs = np.load('try22/test_accs.npy')




train_loss_data = pd.DataFrame(data = train_loss)        #pandas内置函数
train_accs_data = pd.DataFrame(data = train_accs)
test_accs_data = pd.DataFrame(data = test_accs)
train_loss_data.to_csv('try22/train_loss.csv')
train_accs_data.to_csv('try22/train_accs.csv')
test_accs_data.to_csv('try22/test_accs.csv')
