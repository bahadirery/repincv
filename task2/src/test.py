from selectors import EpollSelector
import torch
import numpy as np
import cv2
import pandas as pd
import os
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from model import build_model
from dataset import ImageDataset

from sklearn.metrics import precision_score, recall_score, f1_score
# Constants and other configurations.
BATCH_SIZE = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 256
NUM_WORKERS = 4
CLASS_NAMES = ['Crack Detected', 'Crack Undetected']

def denormalize(
    x, 
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
):
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)

def save_test_results(tensor, target, output_class, counter):
    """
    This function will save a few test images along with the 
    ground truth label and predicted label annotated on the image.

    :param tensor: The image tensor.
    :param target: The ground truth class number.
    :param output_class: The predicted class number.
    :param counter: The test image number.
    """
    image = denormalize(tensor).cpu()
    image = image.squeeze(0).permute((1, 2, 0)).numpy()
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gt = target.cpu().numpy()
    cv2.putText(
        image, f"GT: {CLASS_NAMES[int(gt)]}", 
        (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, (0, 255, 0), 2, cv2.LINE_AA
    )
    if output_class == gt:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    cv2.putText(
        image, f"Pred: {CLASS_NAMES[int(output_class)]}", 
        (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, color, 2, cv2.LINE_AA
    )
    cv2.imwrite(
        os.path.join('..', 'outputs', 'test_results', 'test_image_'+str(counter)+'.png'), 
        image*255.
    )

def test(model, testloader, DEVICE):

    """
    Function to test the trained model on the test dataset.

    :param model: The trained model.
    :param testloader: The test data loader.
    :param DEVICE: The computation device.

    Returns:
        predictions_list: List containing all the predicted class numbers.
        ground_truth_list: List containing all the ground truth class numbers.
        acc: The test accuracy.
        precision: The weighted precision of the model on the test dataset.
        recall: The weighted recall of the model on the test dataset.
        f1: The weighted F1-score of the model on the test dataset.
    """
    model.eval()
    print('Testing model')
    predictions_list = []
    ground_truth_list = []
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(image)
            predictions = F.softmax(outputs).cpu().numpy()
            output_class = np.argmax(predictions)
            predictions_list.append(output_class)
            ground_truth_list.append(labels.cpu().numpy())
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()

            if counter % 99 == 0:
                save_test_results(image, labels, output_class, counter)

    acc = 100. * (test_running_correct / len(testloader.dataset))
    
    # Calculate precision, recall and F1-score
    precision = precision_score(ground_truth_list, predictions_list, average='weighted')
    recall = recall_score(ground_truth_list, predictions_list, average='weighted')
    f1 = f1_score(ground_truth_list, predictions_list, average='weighted')
    
    return predictions_list, ground_truth_list, acc, precision, recall, f1

if __name__ == '__main__':
    df = pd.read_csv(os.path.join('..', 'input', 'test.csv'))
    X = df.image_path.values # Image paths.
    y = df.target.values # Targets
    dataset_test = ImageDataset(X, y, tfms=0)

    test_loader = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    checkpoint = torch.load(os.path.join('..', 'outputs', 'model.pth'))
    # Load the model.
    model = build_model(
        pretrained=False,
        fine_tune=False, 
        num_classes=2
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    predictions_list, ground_truth_list, acc = test(
        model, test_loader, DEVICE
    )
    print(f"Test accuracy: {acc:.3f}%")