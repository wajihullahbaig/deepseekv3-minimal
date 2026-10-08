import pandas as pd
import pytest
import torch

from conftest import YT_CSV
from data.chunking import TextChunkDataset, chunk_token_ids, collate_fn, make_loaders, split_by_source
from data.preprocessing import clean_wikipedia_text
from data.wikipedia_dataloaders import preprocess_and_chunk_dataset
from data.youtube_dataloaders import preprocess_and_chunk_dataframe


def _words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_dataset_item_is_next_token_shifted_and_padded(stub_tokenizer):
    chunk = list(range(2, 22))                                    # 20 ids -> 19 (input, target) pairs
    item = TextChunkDataset([chunk], stub_tokenizer, max_length=32)[0]
    assert item["input_ids"].shape == item["output_ids"].shape == item["attention_mask"].shape == (32,)
    assert item["input_ids"][:19].tolist() == chunk[:-1] and (item["input_ids"][19:] == 0).all()
    assert item["output_ids"][:19].tolist() == chunk[1:] and (item["output_ids"][19:] == 0).all()
    assert item["attention_mask"].tolist() == [1] * 19 + [0] * 13


def test_full_chunk_last_target_is_the_true_next_token(stub_tokenizer):
    chunk = list(range(2, 35))                                    # max_length + 1 ids
    item = TextChunkDataset([chunk], stub_tokenizer, max_length=32)[0]
    assert item["input_ids"].tolist() == chunk[:-1]
    assert item["output_ids"].tolist() == chunk[1:]
    assert item["output_ids"][-1].item() == 34 != stub_tokenizer.eos_token_id


def test_chunker_keeps_short_texts_and_drops_tiny_ones():
    assert chunk_token_ids(list(range(40)), max_length=128, stride=32, min_length=5) == [list(range(40))]
    assert chunk_token_ids([7, 8, 9], max_length=128, stride=32, min_length=5) == []
    assert chunk_token_ids([7], max_length=128, stride=32, min_length=1) == []       # no pair to learn from


def test_chunkers_produce_overlapping_windows(stub_tokenizer):
    text = _words(300)                                            # 301 ids incl. EOS
    df = pd.DataFrame({"Comment": [text]})
    yt, yt_owners = preprocess_and_chunk_dataframe(df, stub_tokenizer, max_length=128, stride=32, min_length=5, input_column="Comment")
    wiki, _ = preprocess_and_chunk_dataset([{"text": text}], stub_tokenizer, max_length=128, stride=32, min_length=5)
    assert yt == wiki
    assert len(yt) == (301 - 129) // 32 + 1
    assert all(len(c) == 129 for c in yt) and set(yt_owners) == {0}
    assert yt[0][32:] == yt[1][:97]                               # neighbours overlap by 97 ids


def test_split_by_source_keeps_each_text_in_one_split():
    owners = [i // 3 for i in range(60)]                          # 20 texts x 3 chunks
    chunks = [[o, i] for i, o in enumerate(owners)]
    train, val, test = split_by_source(chunks, owners, seed=1)
    assert len(train) + len(val) + len(test) == 60
    owner_sets = [{c[0] for c in s} for s in (train, val, test)]
    assert owner_sets[0].isdisjoint(owner_sets[1]) and owner_sets[0].isdisjoint(owner_sets[2]) and owner_sets[1].isdisjoint(owner_sets[2])
    assert len(owner_sets[0]) == 14 and len(owner_sets[1]) == 3 and len(owner_sets[2]) == 3


def test_augmentation_only_applies_to_the_train_loader(stub_tokenizer):
    splits = ([list(range(2, 20))] * 4, [list(range(2, 20))] * 2, [list(range(2, 20))] * 2)
    train, val, test = make_loaders(splits, stub_tokenizer, max_length=32, batch_size=2, use_augmentations=True,
                                    num_workers=0)
    assert train.dataset.use_augmentations and not val.dataset.use_augmentations and not test.dataset.use_augmentations
    ids, mask, out = next(iter(train))
    assert ids.shape == mask.shape == out.shape == (2, 32)


def test_collate_stacks_to_batch(stub_tokenizer):
    ds = TextChunkDataset([list(range(2, 12)), list(range(2, 30))], stub_tokenizer, max_length=32)
    ids, mask, out = collate_fn([ds[0], ds[1]])
    assert ids.shape == mask.shape == out.shape == (2, 32)
    assert ids.dtype == out.dtype == torch.long


def test_clean_wikipedia_text_strips_markup():
    assert clean_wikipedia_text("Hi [1] there {{cite}} [[link]]   x") == "Hi there x"


@pytest.mark.slow
def test_youtube_corpus_coverage(mt5_tokenizer):
    if not YT_CSV.exists():
        pytest.skip(f"YouTube CSV not found at {YT_CSV}")
    texts = pd.read_csv(YT_CSV)["Comment"].dropna().tolist()[:18000]
    kept = sum(bool(chunk_token_ids(mt5_tokenizer.encode(t, add_special_tokens=True), 128, 32, 5)) for t in texts)
    print(f"\ncomments: {len(texts)}  producing training data: {kept} ({100 * kept / len(texts):.1f}%)")
    assert kept >= 0.95 * len(texts)
