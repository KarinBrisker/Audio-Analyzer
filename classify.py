from typing import List, Optional
import torch
import torch.nn.functional as F
import whisper
from tqdm import tqdm
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.model import Whisper
from whisper.tokenizer import Tokenizer, get_tokenizer

CLASS_NAMES_PATH = "resources/class_names.txt"
AUDIO_DATA_PATH = "resources/samples/nunu_short.mp3"


def read_class_names(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f]


@torch.no_grad()
def calculate_average_logprobs(model: Whisper, audio_features: torch.Tensor, class_names: List[str],
                               tokenizer: Tokenizer) -> torch.Tensor:
    initial_tokens = (torch.tensor(tokenizer.sot_sequence_including_notimestamps).unsqueeze(0).to(model.device))
    eot_token = torch.tensor([tokenizer.eot]).unsqueeze(0).to(model.device)

    average_logprobs = torch.zeros(len(class_names))
    for i, class_name in enumerate(class_names):
        class_name_tokens = (
            torch.tensor(tokenizer.encode(" " + class_name)).unsqueeze(0).to(model.device)
        )
        input_tokens = torch.cat([initial_tokens, class_name_tokens, eot_token], dim=1)

        logits = model.logits(input_tokens, audio_features)  # (1, T, V)
        logprobs = F.log_softmax(logits, dim=-1).squeeze(0)  # (T, V)
        logprobs = logprobs[len(tokenizer.sot_sequence_including_notimestamps) - 1 : -1]  # (T', V)
        logprobs = torch.gather(logprobs, dim=-1, index=class_name_tokens.view(-1, 1))  # (T', 1)
        average_logprob = logprobs.mean().item()
        average_logprobs[i] = average_logprob
    return average_logprobs


@torch.no_grad()
def calculate_audio_features(audio_path: Optional[str], model: Whisper) -> torch.Tensor:
    mel = log_mel_spectrogram(audio_path)
    segment = pad_or_trim(mel, N_FRAMES).to(model.device)
    return model.embed_audio(segment.unsqueeze(0))



def classify(model: Whisper, audio_path: str, class_names: List[str], tokenizer: Tokenizer,
             verbose: bool = False) -> str:
    audio_features = calculate_audio_features(audio_path, model)
    average_logprobs = calculate_average_logprobs(model=model, audio_features=audio_features,
                                                  class_names=class_names, tokenizer=tokenizer)
    sorted_indices = sorted(range(len(class_names)), key=lambda i: average_logprobs[i], reverse=True)

    if verbose:
        tqdm.write("  Average log probabilities for each class:")
        for i in sorted_indices:
            tqdm.write(f"    {class_names[i]}: {average_logprobs[i]:.3f}")
    return class_names[sorted_indices[0]]


def main():
    language = "he"
    records = [AUDIO_DATA_PATH]
    class_names = read_class_names(CLASS_NAMES_PATH)
    tokenizer = get_tokenizer(multilingual=True, language=language)
    model = whisper.load_model("large")

    results = []
    for record in tqdm(records):
        tqdm.write(f"processing {record}")
        result = classify(model=model, audio_path=record, class_names=class_names, tokenizer=tokenizer)
        results.append(result)
        tqdm.write(f"  predicted: {result}")


if __name__ == "__main__":
    main()

# python classify.py --audio <audio_file_path> --class_names <class_names_file_path>
