import onnx

model = onnx.load("AUOD_test.onnx")

print("输入节点信息：")
for input in model.graph.input:
    print(f"  名称: {input.name}")
    print(f"  形状: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")

print("\n输出节点信息：")
for output in model.graph.output:
    print(f"  名称: {output.name}")
    print(f"  形状: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")