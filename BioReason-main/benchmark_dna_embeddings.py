import argparse
import time
from statistics import mean

import torch

from bioreason.models.dna_llm import DNALLMModel


def build_model(text_model_name: str, dna_model_name: str, layer_name: str) -> DNALLMModel:
    model = DNALLMModel(
        text_model_name=text_model_name,
        dna_model_name=dna_model_name,
        dna_is_evo2=True,
        dna_embedding_layer=layer_name,
        text_model_finetune=False,
        dna_model_finetune=False,
    )

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def prepare_inputs(model: DNALLMModel, num_samples: int, seqs_per_sample: int, seq_len: int):
    def make_sequence(length: int) -> str:
        bases = ["A", "T", "C", "G"]
        return "".join(bases[i % 4] for i in range(length))

    batch_dna_sequences = [
        [make_sequence(seq_len + j) for j in range(seqs_per_sample)]
        for _ in range(num_samples)
    ]

    texts = [
        f"<|dna_pad|> Sample {idx}"
        for idx in range(num_samples)
    ]

    processor_inputs = model.processor(
        batch_dna_sequences=batch_dna_sequences,
        text=texts,
        return_tensors="pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    dna_tokenized = processor_inputs["dna_tokenized"]
    batch_idx_map = processor_inputs["batch_idx_map"]

    for key, tensor in dna_tokenized.items():
        if isinstance(tensor, torch.Tensor):
            dna_tokenized[key] = tensor.to(
                device=model.dna_projection.weight.device,
                dtype=torch.int64,
            )

    return dna_tokenized, batch_idx_map, len(batch_dna_sequences)


def process_dna_embeddings_old(
    model: DNALLMModel,
    dna_tokenized,
    batch_idx_map,
    batch_size: int,
):
    """旧版实现：逐序列调用"""
    input_ids = dna_tokenized["input_ids"]
    attention_mask = dna_tokenized["attention_mask"]

    if input_ids.size(0) == 0:
        empty = torch.zeros(
            (0, model.text_hidden_size),
            device=model.dna_projection.weight.device,
            dtype=model.dna_projection.weight.dtype,
        )
        return [empty.clone() for _ in range(batch_size)]

    with torch.no_grad():
        hidden_states_list = []
        for seq_idx in range(len(input_ids)):
            seq_input_ids = input_ids[seq_idx : seq_idx + 1]
            _, embeddings = model.dna_model(
                seq_input_ids,
                return_embeddings=True,
                layer_names=[model.dna_embedding_layer],
            )
            seq_embeddings = embeddings[model.dna_embedding_layer].squeeze(0)
            hidden_states_list.append(seq_embeddings)

        if hidden_states_list:
            hidden_states = torch.stack(hidden_states_list)
        else:
            empty = torch.zeros(
                (0, model.text_hidden_size),
                device=model.dna_projection.weight.device,
                dtype=model.dna_projection.weight.dtype,
            )
            return [empty.clone() for _ in range(batch_size)]

    hidden_states = hidden_states.to(
        device=model.dna_projection.weight.device,
        dtype=model.dna_projection.weight.dtype,
    )
    projected_states = model.dna_projection(hidden_states)

    result = [[] for _ in range(batch_size)]
    for seq_idx, batch_idx in enumerate(batch_idx_map):
        valid_length = attention_mask[seq_idx].sum().item()
        seq_embedding = projected_states[seq_idx, :valid_length]
        result[batch_idx].append(seq_embedding)

    final_result = []
    for embeddings in result:
        if embeddings:
            final_result.append(torch.cat(embeddings, dim=0))
        else:
            final_result.append(
                torch.zeros(
                    (0, model.text_hidden_size),
                    device=model.dna_projection.weight.device,
                    dtype=model.dna_projection.weight.dtype,
                )
            )

    return final_result


def process_dna_embeddings_new(
    model: DNALLMModel,
    dna_tokenized,
    batch_idx_map,
    batch_size: int,
):
    """新版实现：批量调用"""
    return model.process_dna_embeddings(dna_tokenized, batch_idx_map, batch_size)


def benchmark(fn, iterations: int, warmup: int):
    """运行基准测试"""
    durations: list[float] = []

    # 预热
    for _ in range(warmup):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 正式测试
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)

    return mean(durations)


def main():
    parser = argparse.ArgumentParser(description="Benchmark DNA embedding processing.")
    parser.add_argument("--text_model", required=True, help="文本模型名称，例如 Qwen/Qwen3-0.5B")
    parser.add_argument("--dna_model", required=True, help="Evo2 模型名称")
    parser.add_argument("--layer", required=True, help="需要提取的 Evo2 层名")
    parser.add_argument("--samples", type=int, default=4, help="批大小（样本数）")
    parser.add_argument("--seqs_per_sample", type=int, default=2, help="每个样本含 DNA 序列数")
    parser.add_argument("--seq_len", type=int, default=512, help="每条 DNA 序列长度")
    parser.add_argument("--iters", type=int, default=10, help="正式计时迭代次数")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")

    args = parser.parse_args()

    print("Loading model...")
    model = build_model(args.text_model, args.dna_model, args.layer)

    print("Preparing inputs...")
    dna_tokenized, batch_idx_map, batch_size = prepare_inputs(
        model,
        num_samples=args.samples,
        seqs_per_sample=args.seqs_per_sample,
        seq_len=args.seq_len,
    )

    print("Running sanity check...")
    old_res = process_dna_embeddings_old(model, dna_tokenized, batch_idx_map, batch_size)
    new_res = process_dna_embeddings_new(model, dna_tokenized, batch_idx_map, batch_size)

    assert len(old_res) == len(new_res) == batch_size, "输出样本数量不一致"
    for idx, (old_tensor, new_tensor) in enumerate(zip(old_res, new_res)):
        if old_tensor.shape != new_tensor.shape:
            raise RuntimeError(
                f"样本 {idx} 的输出形状不一致: old={old_tensor.shape}, new={new_tensor.shape}"
            )
    print("Shape check passed!")

    print("Benchmarking old implementation...")
    old_time = benchmark(
        lambda: process_dna_embeddings_old(model, dna_tokenized, batch_idx_map, batch_size),
        iterations=args.iters,
        warmup=args.warmup,
    )

    print("Benchmarking new implementation...")
    new_time = benchmark(
        lambda: process_dna_embeddings_new(model, dna_tokenized, batch_idx_map, batch_size),
        iterations=args.iters,
        warmup=args.warmup,
    )

    print("\n=== Results ===")
    print(f"旧版平均耗时：{old_time:.6f} 秒")
    print(f"新版平均耗时：{new_time:.6f} 秒")
    if old_time > 0:
        speedup = old_time / new_time
        print(f"加速比（旧 / 新）：{speedup:.2f}x")
    else:
        print("旧版耗时为 0，无法计算加速比")


if __name__ == "__main__":
    main()

