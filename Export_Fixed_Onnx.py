import torch
from ultralytics import YOLO

def export_fixed_onnx(pt_path, onnx_path, imgsz=[224, 224]):
    """导出固定维度的 ONNX 模型"""
    # 加载模型
    model = YOLO(pt_path)
    model = model.model.cpu()  # 确保在 CPU 上运行
    
    # 创建固定尺寸的输入张量
    dummy_input = torch.randn(1, 3, imgsz[0], imgsz[1], requires_grad=True)
    
    # 导出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['images'],  # 明确指定输入名称
        output_names=['output0'],  # 明确指定输出名称
        dynamic_axes={}  # 空字典表示无动态轴
    )
    
    print(f"✅ 模型已导出为: {onnx_path}")
    
    # 验证导出的模型
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX 模型验证通过")
    
    # 打印输入输出形状
    print("\n验证输入输出形状:")
    for input in onnx_model.graph.input:
        print(f"  输入名称: {input.name}, 形状: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
    
    for output in onnx_model.graph.output:
        print(f"  输出名称: {output.name}, 形状: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")

if __name__ == "__main__":
    pt_path = "/home/AUOD.pt"
    onnx_path = "/home/AUOD.onnx"
    export_fixed_onnx(pt_path, onnx_path, imgsz=[224, 224])