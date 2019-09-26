import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms, utils
import torch.optim as optim
from models.vgg_like_model import vgg_like_structure,init_weights
import torch.nn.functional as F

def train(model,train_data,optimize,cuda):
    model.train()
    for data,target in train_data:
        if cuda:
            data = data.cuda()
            target = target.cuda()
        optimize.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimize.step()
        print('Batch loss: {}'.format(loss.item()))

def test(model,test_data,cuda):
    correct = 0
    model.eval()
    with torch.no_grad():
        for data,target in test_data:
            if cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            pred = output.argmax(dim=1,keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    print('Test set:  {}  accuracy'.format(correct / len(test_data.dataset) * 100.))
        


def run(model,train_data,test_data,optimize,cuda=False,n_epochs=3):
    for i in range(n_epochs):
        print('Epoch {} Starting'.format(i + 1))
        train(model,train_data,optimize,cuda)
        test(model,test_data,cuda)
        torch.save(model.state_dict(), 'models/mnist_train_epoch_{}.pt'.format(i + 1))



if __name__ == '__main__':
    trans = transforms.Compose([transforms.ToTensor()]) #I'm skipping normalizing because these images are small and converge fast anyway
    mnist_train_dataset = datasets.MNIST('datasets',train=True,download=True,transform=trans)
    mnist_test_dataset = datasets.MNIST('datasets',train=False,download=True,transform=trans)


    train_loader = DataLoader(mnist_train_dataset,batch_size=100,shuffle=True)
    test_loader = DataLoader(mnist_test_dataset,batch_size=1000,shuffle=True)

    cuda_check = torch.cuda.is_available()

    model = vgg_like_structure

    if cuda_check:
        model = vgg_like_structure.cuda()

    init_weights(model)

    optimize = optim.Adam(model.parameters(), lr=0.001)  #I tend to use Adam, can use SGD as suggested, Adam is just because it's usually how Caffe models are done

    run(model,train_loader,test_loader,optimize,cuda_check)

