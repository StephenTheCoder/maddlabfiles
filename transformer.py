from transformers import AutoImageProcessor, Dinov2ForImageClassification
import torch
from PIL import Image

# Load the pre-trained model and image processor
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base")

# Set the model to evaluation mode
model.eval()