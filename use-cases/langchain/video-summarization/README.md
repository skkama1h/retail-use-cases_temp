# Video Summarization using OpenVINO, LangChain, and VLMs + LLMs

## Installation
* Install Intel Client GPU, Conda, and Set Up Python Environment
```
# Validated on Ubuntu 24.04 and 22.04
bash install.sh
```

## Stage 1: Create Video Chunk Summaries
This  pipeline uses OpenVINO-GenAI, Langchain, and MiniCPM-V-2_6 for generating video chunk summaries.

### Convert and Save OpenVINO Optimized MiniCPM-V-2_6
First, follow the steps on the [MiniCPM-V-2_6 HuggingFace Page](https://huggingface.co/openbmb/MiniCPM-V-2_6) to gain access to the model. For more information on user access tokens for access to gated models see [here](https://huggingface.co/docs/hub/en/security-tokens).

Next, convert and save the optimized model.
```
optimum-cli export openvino -m openbmb/MiniCPM-V-2_6 --trust-remote-code --weight-format int8 MiniCPM_INT8 # int4 also available 
```

### Run Video Summarization pipeline
Summarize a video using `video_summarizer.py`. For example, start by downloading the following video:
 
```
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4
```

Now create summaries of that video using `video_summarizer.py`. This script chunks the video and outputs chunk summaries to 
`output_dir/<input video name>.json`
```
input_video="one-by-one-person-detection.mp4"
python video_summarizer.py $input_video MiniCPM_INT8/ -d "GPU" -r 480 270 
```

## Stage 2: Merge Chunk Summaries and Assign Anomaly Score

### Create OpenVINO optimized model
By default, the script uses `llmware/llama-3.2-3b-instruct-ov` model which does not require any conversion. 
Please skip this step if you are using the default model.

If you want to use a different model (ensure you have gained access via HuggingFace if the model requires it), 
convert and then save the optimized model using the following command.
For example: If you wanted to use `meta-llama/Llama-3.2-3B-Instruct`, follow the instruction on [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) to gain access to the model.
Next, perform the OpenVINO conversion using the below command:
```
model_id="meta-llama/Llama-3.2-3B-Instruct"
optimum-cli export openvino -m $model_id --trust-remote-code --weight-format int8 llama_32_3b_int8 # int4 also available 
```

Finally, create one Merged Summary with an Anomaly Score indicating the likelihood of an anomaly/suspicious activity:
```
output_dir=./output
input_video="one-by-one-person-detection.mp4"
summaries_file=$(basename "input_video" .mp4).json
python summary_merger.py "$output_dir/$summaries_file" -d "GPU"
```

**NOTE:** If you are using a non-default model which you optimized using the `optimum-cli` command, 
specify the model path using the `-m` flag:
```
python summary_merger.py "$output_dir/$summaries_file" -d "GPU" -m llama_32_3b_int8
```
View the Chunk Summaries, Merged Summaries and the Anomaly Score in the `output_dir/<input video name>.json` file.

## Run Stage 1 and Stage 2 pipeline using a single script
You can run the entire pipeline using a single script. This script will run the video summarization pipeline 
and then merge the chunk summaries and assign an anomaly score.

Please open the `driver.sh` and edit three variables:
* `input_video` - path to the video file

If you are using a different model for the VLM/LLM, **only then edit the following variables**:
* `vlm_model` - path to the MiniCPM openVINO optimized model 
* `llm_model` - path to the Llama openVINO optimized model 

Then run the script: `./driver.sh`