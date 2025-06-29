import pandas as pd
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification
from transformers import TrainingArguments, Trainer

print("🔥 TrainingArguments from:", TrainingArguments.__module__)

# 트위터 데이터 셋을 불러온다
dataset = load_dataset("mteb/tweet_sentiment_extraction")

# 트레이닝 데이터 셋을 데이타 프레임 형태로 변환한다 
df = pd.DataFrame(dataset['train'])

# 토크나이저를 설정한다
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 이전에 설정한 토크나이저를 이용해서 데이터를 토크나이즈하는 함수를 정의한다
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 트위터 데이터 셋을 토크나이즈한다 
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 토크나이즈 된 데이터 셋을 트레인 및 테스트 데이터 셋으로 나눈다
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# GPT2를 3 종류의 라벨로 분류하는 모델을 만든다
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)

# 평가 메트릭을 불러온다
metric = evaluate.load("hyperml/balanced_accuracy")

# 가장 확률이 높은 라벨을 예측하는 함수를 정의한다 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
    
# 트레이닝에 필요한 설정을 정의한다
# TrainingArguments 클래스를 사용하여 객체를 생성한다
training_args = TrainingArguments(
    output_dir="test_trainer", # 모델을 저장할 공간 
    evaluation_strategy="epoch", # 전체 데이터 셋을 돌리고 평가하는 전략을 택한다
    per_device_train_batch_size=1, # Reduce batch size here
    per_device_eval_batch_size=1, # Optionally, reduce for evaluation as well
    gradient_accumulation_steps=4 # 4 문장 마다 가중치를 업데이트 합니다 
)

# 트레이닝 하는 클레스를 이용해서 객체를 생성한다
trainer = Trainer(
    model=model, # 3 종류의 라벨로 분류하는 모델을 만든다
    args=training_args, # 트레이닝에 필요한 설정 Arguments를 이용해서 객체를 생성한다
    train_dataset=small_train_dataset, # 
    eval_dataset=small_test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()