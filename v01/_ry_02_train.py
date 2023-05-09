'''
_ry_02_train.py
'''

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# %%
from torchaudio.datasets import SPEECHCOMMANDS
import os

data_path= "/_ryDatasets"
# check if the dircetory exists, if not, make it
if not os.path.isdir(data_path):
    os.mkdir(data_path)


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__(data_path, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set= SubsetSC("training")
test_set=  SubsetSC("testing")
val_set=   SubsetSC("validation")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

from ryModels import theLabels, label_to_index, index_to_label

labels= theLabels


# %%
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 1024 #1000 #1024 #256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

#%%

#%%
#
# Move it out to another file, for both training and testing
#

from ryModels import ryM10 as ryM

model= ryM(in_chs= 1, out_cls=35)

# ryM2 Test@epoch= 15, acc=【0.8678】, [9550/11005]
# Number of parameters: 590_563

# ryM, Test@epoch= 13, acc=【0.8706】, [9581/11005]

# Train@epoch= 15, Loss: 0.205410
# ryM,  Test@epoch= 15, acc= 0.8642, [9510/11005]

# ryM,  Test@epoch= 15, acc= 0.8531, [9388/11005]
# M6,   Test Epoch: 24  Accuracy: 9362/11005 (85%)
# M5_1, Test Epoch: 21  Accuracy: 8905/11005 (81%)

#%%
#%%
# load the weights
# check the availability of "model.pt"
#'''
model_fn= 'ryM.pt'

#if os.path.isfile(model_fn):
#    model.load_state_dict(torch.load(model_fn))
#'''
#%%

model.to(device)
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_parameters(model)
print("Number of parameters: %s" % n)


# %%
# using CrossEntropyLoss as our loss function
# criterion= nn.CrossEntropyLoss()

# using negative log likelihood loss as the loss function
loss_fn=   nn.NLLLoss()

optimizer= optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler= optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  
# reduce the learning after 10 epochs by a factor of 10

# %%
def train(model, epoch, log_interval, lossL= []):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data=   data.to(device)
        target= target.to(device)

        output= model(data)

        loss= loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossL += [loss.item()]

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTrain@{epoch= }, Loss: {lossL[-1]:.6f}")
    return lossL


# %%
def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def test(model, epoch=1, test_or_val= 'val'):
    model.eval()
    correct = 0

    #loader= (test_loader if test_or_val=='test' else 
    #         val_loader)
    
    if test_or_val=='test':
        loader= test_loader
    elif test_or_val=='val':
        loader= val_loader
    elif test_or_val=='train':
        loader= train_loader
    else:
        raise ValueError("test_or_val must be 'test' or 'val' or 'train'")

    for data, target in loader:

        data=   data.to(device)
        target= target.to(device)

        # apply transform and model on whole batch directly on device
        # data = transform(data)
        output= model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        #pbar.update(pbar_update)

    acc= correct/len(loader.dataset)
    print(f"\nTest@{epoch= }, acc=【{acc:.4f}】, [{correct}/{len(loader.dataset)}]\n")
    
    return acc

# %%
log_interval=  100
test_interval=   5
n_epoch=        30

pbar_update = 1 / (len(train_loader) + len(test_loader))
lossL= []
accL= []

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):

        lossL= train(model, epoch, log_interval, lossL)
        
        if (epoch % test_interval == 0) or (epoch == 1):
            acc= test(model, epoch)
            accL += [acc]
            if acc >= max(accL):
                torch.save(model.state_dict(), model_fn)
            
        scheduler.step()
#%%
# finally, test the model on the test set
# save the trained model
acc= test(model)
accL += [acc]
if acc >= max(accL):
    torch.save(model.state_dict(), model_fn)


# %%
'''
# plot the loss
plt.figure()
plt.subplot(1,2,1)
plt.plot(lossL)
plt.xlabel("batch")
plt.ylabel("loss")
#plt.show()

# plot the accuracy
plt.subplot(1,2,2)
plt.plot(accL)
plt.xlabel(f"epoch/{test_interval}")
plt.ylabel("Val accuracy")
plt.show()
'''
# %%

# initialize the model
mdl= ryM(in_chs= 1, out_cls=35)
mdl.to(device)

# load the weights

# get the directory of the current file
# dir= os.path.dirname(os.path.abspath(__file__))
# get the path to the file
path= model_fn #os.path.join(dir, 'model.pt')

mdl.load_state_dict(torch.load(path))

# %%
# test the model
mdl.eval()

acc_val= test(mdl, test_or_val='val')
print(f'Val accuracy: {acc_val:.4f}')

acc_test= test(mdl, test_or_val='test')
print(f'Test accuracy: {acc_test:.4f}')


# %%
'''
Test@epoch= 1, acc=【0.8835】, [8818/9981]
Val accuracy: 0.8835

Test@epoch= 1, acc=【0.8722】, [9599/11005]
Test accuracy: 0.8722
'''
#%%
#%%
#### just for fun, test the model on the training set
acc_train= test(mdl, test_or_val='train')
print(f'Train accuracy: {acc_train:.4f}')
'''
Test@epoch= 1, acc=【0.9596】, [81419/84843]
Train accuracy: 0.9596
'''

# %%
