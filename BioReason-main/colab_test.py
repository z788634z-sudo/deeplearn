"""
Google Colab 测试脚本
用于在 Colab 中测试 BioReason 批量处理优化

使用方法：
1. 在 Colab 中上传此文件
2. 上传 bioreason/models/dna_llm.py（包含你的优化）
3. 运行此脚本
"""

# 安装依赖
import subprocess
import sys

def install_dependencies():
    """安装必要的依赖"""
    print("安装依赖...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                          "torch", "torchvision", "transformers", "accelerate"])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "evo2"])
        print("✓ evo2 安装成功")
    except:
        print("⚠ evo2 安装失败，可能需要手动安装")

# 检查 GPU
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ 未检测到 GPU，将使用 CPU（会很慢）")

# 导入模块
try:
    from bioreason.models.dna_llm import DNALLMModel
    print("✓ 模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("请确保已上传所有必要的文件到 Colab")
    sys.exit(1)

# 简化测试
def simple_test():
    """简单的功能测试"""
    print("\n=== 简单功能测试 ===")
    
    # 使用较小的模型
    print("加载模型（这可能需要几分钟）...")
    try:
        model = DNALLMModel(
            text_model_name="Qwen/Qwen3-0.5B",
            dna_model_name="evo2-small",  # 根据实际情况调整
            dna_is_evo2=True,
            dna_embedding_layer="encoder.layer.10",
            text_model_finetune=False,
            dna_model_finetune=False,
        )
        model.eval()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"✓ 模型已加载到 {device}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False
    
    # 准备测试数据
    print("\n准备测试数据...")
    dna_seqs = [
        ["ATCGATCGATCG", "GGGAAACCC"],
        ["TTTTCCCCGGGG"]
    ]
    texts = ["<|dna_pad|> Sample 1", "<|dna_pad|> Sample 2"]
    
    processor_inputs = model.processor(
        batch_dna_sequences=dna_seqs,
        text=texts,
        return_tensors="pt",
        device=device,
    )
    
    print(f"✓ 数据准备完成")
    print(f"  - 批大小: {len(dna_seqs)}")
    print(f"  - DNA 序列数: {len(processor_inputs['dna_tokenized']['input_ids'])}")
    
    # 测试批量处理
    print("\n测试批量处理...")
    try:
        result = model.process_dna_embeddings(
            processor_inputs["dna_tokenized"],
            processor_inputs["batch_idx_map"],
            batch_size=2
        )
        
        print(f"✓ 批量处理成功！")
        print(f"  - 返回结果数: {len(result)}")
        for i, tensor in enumerate(result):
            print(f"  - 样本 {i} 形状: {tensor.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 批量处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 安装依赖
    install_dependencies()
    
    # 运行测试
    success = simple_test()
    
    if success:
        print("\n" + "="*50)
        print("✓ 所有测试通过！批量处理功能正常工作。")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("✗ 测试失败，请检查错误信息。")
        print("="*50)

