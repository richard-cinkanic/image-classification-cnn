import torch
from torch.utils.data import DataLoader, random_split
from dataset import ImagesDataset
from architecture import model

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 32
num_epochs = 30
learning_rate = 0.001
image_dir = "training_data"

dataset = ImagesDataset(image_dir=image_dir)
train_size, val_size = int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_network(network, train_loader, optimizer, loss_function):
    network.train()
    train_loss = 0.0
    for data in train_loader:
        input, target = data[0], data[1]
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(input)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def evaluate_network(network, eval_loader, loss_function):
    network.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for data in eval_loader:
            input, target = data[0], data[1]
            input, target = input.to(device), target.to(device)
            output = network(input)
            loss = loss_function(output, target)
            eval_loss += loss.item()
    return eval_loss / len(eval_loader)

def early_stopping(validation_loss, min_eval_loss, counter):
    counter = 0 if validation_loss < min_eval_loss else counter + 1
    min_eval_loss = min(min_eval_loss, validation_loss)
    return min_eval_loss, counter

def training_loop(
    network: torch.nn.Module,
    train_data: torch.utils.data.Dataset,
    eval_data: torch.utils.data.Dataset,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_tolerance: int = 7,
) -> tuple[list, list, list]:
    
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    loss_function = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    eval_losses = []
    
    min_eval_loss = None
    counter = 0
    
    for epoch in range(num_epochs):
        train_loss = train_network(network, train_loader, optimizer, loss_function)
        eval_loss = evaluate_network(network, eval_loader, loss_function)
        
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}')
        
        if epoch == 0:
            min_eval_loss = eval_loss
            
        min_eval_loss, counter = early_stopping(eval_loss, min_eval_loss, counter)
        
        if counter == max_tolerance:
            print("Early stopping")
            break
        
    return train_losses, eval_losses

train_losses, eval_losses = training_loop(
    network=model,
    train_data=train_dataset,
    eval_data=val_dataset,
    num_epochs=num_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
)

torch.save(model.state_dict(), 'model.pth')
