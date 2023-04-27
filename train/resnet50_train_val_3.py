######### Set workding dir, device and random seed ############                          
# working directory
wd = "/gpfs/scratch/hj2470/DL_project_2/physionet.org/files/mimic-cxr-jpg/2.0.0/"
import os
os.chdir(wd)
print(f"Working dir: {wd}")

# device
import torch
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU   
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

print("Device being used: %s" %device)

# random seed
import torch
import numpy as np
import random 

random_seed = 2023
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)



###### HYPERPARAMETERS ########                                                           
batch_size = 64
epochs = 24
lr = 1e-3
MODEL_NAME = "resnet50_pretrainedT_reqgradF_3_2.pt"

print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Learning rate: {lr}")
print(f"Model name: {MODEL_NAME}")



####### Load datasets #########                                                                                        
import pandas as pd
# subsets
train_path = "train_subset_3.csv"
valid_path = "valid_subset_3.csv"
test_path = "test_subset_3.csv"

train_subset = pd.read_csv(train_path)
valid_subset = pd.read_csv(valid_path)
test_subset = pd.read_csv(test_path)

print(f"Train size: {len(train_subset)}")
print(f"Valid size: {len(valid_subset)}")



####### Chest x-ray dataset class #######                                                                              
import pandas as pd
import os
from skimage import io
import torch
from skimage import color
from torch.utils.data import Dataset, DataLoader

class ChestXrayDataset(Dataset):
    """Chest X-ray dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                str(self.data_frame.iloc[idx, 0]))
        
        img_name = img_name+".jpg"
        
        #### TODO: Read in image using io
        image = io.imread(img_name)
        
        ###### TODO: normalise the image
        image = color.rgb2gray(image)
        #image = image/image.max()
        image = torch.from_numpy(np.repeat(image[None,...],3,axis=0))

        ###### TODO: return dictionary of image and corresponding label
        # torch.from_numpy(np.array(split_label.iloc[idx, 4:]).astype(int))
        label = torch.from_numpy(np.array(self.data_frame.iloc[idx, 4:]).astype(int))
        
        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 'label': label}
        
        return sample



######## Define transform ########                                                                                     
from torchvision import transforms

#train_transform = transforms.Compose([
#        transforms.ToPILImage(),
#        transforms.Resize((224,224)),
#        transforms.ToTensor()])


train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(896),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(896),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



####### Create transformed dataset and dataloader ########                                                             
from torch.utils.data import Dataset, DataLoader

root_dir = "/gpfs/scratch/hj2470/DL_project_2/physionet.org/files/mimic-cxr-jpg/2.0.0/files/images"

transformed_dataset = {'train': ChestXrayDataset(csv_file=train_path, root_dir=root_dir, 
                                                 transform=train_transform),
                       'validate': ChestXrayDataset(csv_file=valid_path, root_dir=root_dir, 
                                                    transform=test_transform),
                       'test': ChestXrayDataset(csv_file=test_path, root_dir=root_dir, 
                                                transform=test_transform)
}

train_dataloader = DataLoader(transformed_dataset['train'], batch_size=batch_size,
                              shuffle=True, drop_last = False)
valid_dataloader = DataLoader(transformed_dataset['validate'], batch_size=batch_size,
                              shuffle=True, drop_last = False)
test_dataloader = DataLoader(transformed_dataset['test'], batch_size=batch_size,
                              shuffle=True, drop_last = False)

print(f"Input image shape: {next(iter(train_dataloader))['image'].shape}")
print(f"Input label shape: {next(iter(train_dataloader))['label'].shape}")



######## Define model #######                                                                                          
# ResNet50 model
import torchvision.models as models
import torch.nn as nn

def ResNet_model(pretrained, requires_grad):
    model = models.resnet50(progress=True, pretrained=pretrained)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 14 classes in total
    model.fc = nn.Linear(2048, 14)
    return model



####### Define training and evaluating functions #######                                                               ## training function
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    pred_label = []
    true_label = []

    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch['image'].to(device))
        outputs = torch.sigmoid(predictions) 
        loss = criterion(outputs, batch['label'].type(torch.cuda.FloatTensor))
        loss.backward()
        optimizer.step()

        predicted_label = outputs > 0.5
        epoch_acc += (predicted_label.long() == batch['label'].type(torch.cuda.LongTensor)).sum().item()

        predicted_label = predicted_label.long().detach().cpu().numpy().tolist()
        actual_label = batch['label'].detach().cpu().numpy().tolist()
        
        #epoch_loss += loss.detach().cpu().numpy()
        epoch_loss += loss.item() * batch['image'].size(0)    

        pred_label.extend(predicted_label)
        true_label.extend(actual_label)

    return epoch_loss / len(iterator.dataset),  epoch_acc / (len(iterator.dataset)*14), pred_label, true_label


## evaluating function
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    pred_label = []
    true_label = []

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch['image'].to(device))
            outputs = torch.sigmoid(predictions) 
            loss = criterion(outputs, batch['label'].type(torch.cuda.FloatTensor))
            
            predicted_label = outputs > 0.5
            epoch_acc += (predicted_label.long() == batch['label'].type(torch.cuda.LongTensor)).sum().item()

            predicted_label = predicted_label.long().detach().cpu().numpy().tolist()
            actual_label = batch['label'].detach().cpu().numpy().tolist()
        
            #epoch_loss += loss.detach().cpu().numpy()
            epoch_loss += loss.item() * batch['image'].size(0)

            pred_label.extend(predicted_label)
            true_label.extend(actual_label)

    return epoch_loss / len(iterator.dataset), epoch_acc / (len(iterator.dataset)*14), pred_label, true_label



######## Set model and outputs ########                                                                             
import torch.optim as optim

model = ResNet_model(pretrained=True, requires_grad=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

best_valid_loss = float('inf')
avg_loss_dict = {'train': [], 'valid': []}
avg_acc_dict = {'train': [], 'valid': []}
f1_score_dict = {'train': [], 'valid': []}
pred_dict = {'train': [], 'valid': []}
true_dict = {'train': [], 'valid': []}
auc_dict = {'train': [], 'valid': []}



######## Training and validation ########                                                                              
## training and validation
from sklearn.metrics import f1_score, roc_auc_score
from datetime import datetime

start = datetime.now()
print(f"Start time: {start}")

for epoch in range(epochs):
         
    train_loss, train_acc, train_pred, train_true = train(model, train_dataloader, optimizer, criterion)
    valid_loss, valid_acc, valid_pred, valid_true = evaluate(model, valid_dataloader, criterion)

    avg_loss_dict['train'].append(train_loss)
    avg_loss_dict['valid'].append(valid_loss)
    
    avg_acc_dict['train'].append(train_acc)
    avg_acc_dict['valid'].append(valid_acc)

    train_f1 = f1_score(train_true, train_pred, average=None)
    valid_f1 = f1_score(valid_true, valid_pred, average=None)

    f1_score_dict['train'].append(train_f1)
    f1_score_dict['valid'].append(valid_f1)

    train_auc = roc_auc_score(train_true, train_pred, average='macro')
    valid_auc = roc_auc_score(valid_true, valid_pred, average='macro')

    auc_dict['train'].append(train_auc)
    auc_dict['valid'].append(valid_auc)
    
    pred_dict['train'].append(train_pred)
    pred_dict['valid'].append(valid_pred)
    
    true_dict['train'].append(train_true)
    true_dict['valid'].append(valid_true)
    
    # save model with the lowest validation loss
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_NAME)

    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train AUC: {train_auc*100:.2f}% | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. AUC: {valid_auc*100:.2f}% | Val. Acc: {valid_acc*100:.2f}% | ')


end = datetime.now()
print(f"End time: {end}")
print(f"Time taken: {end - start}")



######## Save model outputs #########                                                                                  
import pickle
with open(wd+MODEL_NAME+'.pkl', 'wb') as f:
    pickle.dump([avg_loss_dict, avg_acc_dict, f1_score_dict, pred_dict, true_dict, auc_dict, batch_size, epochs, lr], f)


print("From resnet50_train_val_3.py")
    
