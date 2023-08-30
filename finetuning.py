import datasets
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch


#chat gpt
# run huggingface-cli login to authenticate
# dataset is cached in ~/.cache/huggingface/datasets


def prepare_data(batch, transcripts_dataset, tokenizer):
    # Tokenize transcripts
    batch["input_values"] = tokenizer(transcripts_dataset[batch["id"]]["transcript"], padding="longest",
                                      return_tensors="pt").input_ids
    # Include annotations in training data
    batch["labels"] = batch["annotations"]
    # Convert audio data into format expected by model
    # ...
    return batch

vad_dataset = load_dataset('ivrit-ai/audio-vad', split='train[:5%]')
# vad_dataset.download_and_cache()
# exit()
# transcripts_dataset = load_dataset('ivrit-ai/audio-transcripts')

exit()
# vad_dataset = load_dataset('ivrit-ai/audio-vad')
# vad_dataset.download_and_cache()


model = AutoModelForSpeechSeq2Seq.from_pretrained("whisper/whisper-large")
tokenizer = AutoTokenizer.from_pretrained("whisper/whisper-large")

vad_dataset = vad_dataset.map(lambda batch: prepare_data(batch, transcripts_dataset, tokenizer))

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=vad_dataset['train'],
    eval_dataset=vad_dataset['test'],
)

trainer.train()
