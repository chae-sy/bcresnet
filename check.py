from collections import defaultdict
import random
from utils import SpeechCommandWithSpeaker, PKMTLDataset, Padding
from torchvision import transforms
def check_4tuple_feasibility_for_label(dataset, target_label=1):
    index_by_label = defaultdict(list)
    index_by_speaker = defaultdict(list)

    for idx, (_, label, speaker, _) in enumerate(dataset):
        index_by_label[label].append(idx)
        index_by_speaker[speaker].append(idx)

    valid_count = 0

    for idx in index_by_label[target_label]:
        _, label, speaker, _ = dataset[idx]
        assert label == target_label

        other_labels = list(set(index_by_label.keys()) - {label})
        other_speakers = list(set(index_by_speaker.keys()) - {speaker})

        def has(lbl, spk, exclude=None):
            candidates = set(index_by_label[lbl]) & set(index_by_speaker[spk])
            if exclude is not None:
                candidates.discard(exclude)
            return len(candidates) > 0

        if not other_labels or not other_speakers:
            continue

        if (
            has(label, speaker, exclude=idx) and              # ts-tk
            any(has(lbl, speaker) for lbl in other_labels) and # ts-ntk
            any(has(label, spk) for spk in other_speakers) and # nts-tk
            any(has(lbl, spk) for lbl in other_labels for spk in other_speakers) # nts-ntk
        ):
            valid_count += 1

    print(f"✅ {_label_name(target_label)}: {valid_count} valid 4-tuples found.")

def _label_name(label_id):
    reverse_dict = {0: "_silence_", 1: "_unknown_", 2: "down", 3: "go", 4: "left", 5: "no", 6: "off", 7: "on", 8: "right", 9: "stop", 10: "up", 11: "yes"}
    return reverse_dict.get(label_id, f"label{label_id}")

if __name__:
    case='train'
    base_dir = "./data/speech_commands_v0.02"
    data_dir = f"{base_dir}/{case}_12class"
    noise_dir = f"{base_dir}/_background_noise_"

    transform = transforms.Compose([Padding()])
    train_dataset = PKMTLDataset(SpeechCommandWithSpeaker(data_dir, 2, transform=transform))
    check_4tuple_feasibility_for_label(train_dataset.base, target_label=1)  # Check for _unknown_
    check_4tuple_feasibility_for_label(train_dataset.base, target_label=0)  # Optional: Check _silence_
    
