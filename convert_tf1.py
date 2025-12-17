import tensorflow as tf
import tf2onnx

tf.compat.v1.disable_eager_execution()

with tf.compat.v1.Session() as sess:
    # Load SavedModel
    tf.compat.v1.saved_model.loader.load(sess, ["serve"], "./src/main/resources/model/")

    # Get graph
    graph = tf.compat.v1.get_default_graph()

    # Get tensors
    input_tensor = graph.get_tensor_by_name("serving_default_sequential_input:0")
    output_tensor = graph.get_tensor_by_name("StatefulPartitionedCall:0")

    # Convert
    onnx_graph = tf2onnx.convert.from_graph_def(graph.as_graph_def(), input_names=["serving_default_sequential_input"], output_names=["StatefulPartitionedCall"], opset=13)

    # Save
    with open("./src/main/resources/model/model.onnx", "wb") as f:
        f.write(onnx_graph.SerializeToString())

print("Converted")