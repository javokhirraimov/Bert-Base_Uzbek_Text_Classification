# Bert-Base_Uzbek_Text_Classification

This is a fine-tuned Transformer-based model for classifying Uzbek-language news articles into one of the following categories:

- **Fan va texnika** (Science and Technology)
- **Jahon** (World)
- **Jamiyat** (Public)
- **O'zbekiston** (Uzbekistan)
- **Qonunchilik** (Law)
- **Sport**  (Sport)

Training Progress
Below is a snapshot of the model's training performance during fine-tuning:

Training Metrics

![photo_2025-05-13_13-56-37](https://github.com/user-attachments/assets/cc47b504-f9f2-440d-b92f-981b55f0f260)



How to Use
You can easily load and use this model with the Transformers pipeline:

from transformers import pipeline
import torch

# Load directly from Hugging Face Hub
classifier = pipeline(
    "text-classification",
    model="javokhirraimov/uzbek_news_classifier",
    tokenizer="javokhirraimov/uzbek_news_classifier",
    device=0 if torch.cuda.is_available() else -1
)


##  Training Progress

Below is a snapshot of the model's training performance during fine-tuning:

![Training Metrics](training_metrics.jpg)

## 🧪 How to Use

You can easily load and use this model with the  Transformers `pipeline`:

```python
from transformers import pipeline
import torch

# Load directly from Hugging Face Hub
classifier = pipeline(
    "text-classification",
    model="javokhirraimov/uzbek_news_classifier",
    tokenizer="javokhirraimov/uzbek_news_classifier",
    device=0 if torch.cuda.is_available() else -1
)

# Test it
result = classifier("Toshkentda yangi futbol stadioni qurilmoqda")
print(f"Predicted: {result[0]['label']} (Confidence: {result[0]['score']:.2f})")


Predicted: sport (Confidence: 0.99)
