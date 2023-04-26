# llama_openvino
This sample shows how to implement llama-7b-hf with OpenVINO runtime.

## How to run it?
1. Install the requirements:

    ```$pip install -r requirements.txt```

2. Export the ONNX model from HuggingFace pipeline:

    ```$python export.py -m huggingface_model_path -o onnx_model_path```


    For example: python export.py -m "xxx/llama-7b-hf" -o "./llama.onnx"


    ***please follow the Licence on HuggingFace and get the approval from Meta before downloading llama checkpoints***

3. Convert ONNX model to OpenVINO IR in FP16:

     ```$mo -m onnx_model_path --compress_to_fp16```

4. Run restructured pipeline:

    ```$python generate.py -m openvino_model_path -t tokenizer_path -p prompt_sentence```