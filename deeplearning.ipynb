import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load and preprocess MNIST dataset (handwritten digits for order forms)
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Define a simple neural network
class OrderNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        super(OrderNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # Adds a boost to spot patterns
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)  # Like turning up the right clues
        x = self.layer2(x)
        return x

# Initialize model, loss, and optimizer
model = OrderNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train for a few steps (quick demo)
num_epochs = 1
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if i == 5:  # Stop after 5 batches for speed
            break
        images = images.reshape(-1, 28*28)  # Flatten the digit images
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Real-time use case: Predict a digit from an order form and display the image
sample_image, sample_label = train_dataset[0]  # First image (actual label 5)
sample_image = sample_image.reshape(1, 28*28)
with torch.no_grad():
    prediction = model(sample_image)
    predicted_digit = torch.argmax(prediction).item()

# Display the sample image
plt.figure(figsize=(4, 4))
plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_digit}, Actual: {sample_label}")
plt.axis('off')
plt.show()

print(f"\nPredicted digit on order form: {predicted_digit} (Actual: {sample_label})")
print("Real-Time Use Case: This is like a postal service sorting your handwritten zip codes instantly—useful for delivery jobs!")
