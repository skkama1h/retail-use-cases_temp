#!/bin/bash

source activate-conda.sh
activate_conda
conda activate ovlangvidsumm

# ###################################
# ### Video Pipeline (openvino-genai)
# ###################################

# change the input video path to the video you want to analyze
input_video=$HOME/skk/vids/12_theft.mp4
# change the VLM path to the openVINO optimized dir
vlm_model=$HOME/skk/models/MiniCPM_INT4/

# no change if using defaults
llm_model=llmware/llama-3.2-3b-instruct-ov

# view the video chunks, chunk summaries, merged summaries output files
output_dir=output

# run the video chunk summarizer

python video_summarizer.py "$input_video" "$vlm_model" -d "GPU" -f 32 -c 30 -r 480 270 -t 500 -o \
$output_dir/"$(basename "$input_video" .mp4)".txt -p "As an expert investigator, please analyze this video. Summarize the video, highlighting any shoplifting or suspicious activity. The output must contain the following 3 sections: Overall Summary, Activity Observed, Potential Suspicious Activity. It should be formatted similar to the following example:

**Overall Summary**
Here is a detailed description of the video.

**Activity Observed**
1) Here is a bullet point list of the activities observed. If nothing is observed, say so, and the list should have no more than 10 items.

**Potential Suspicious Activity**
1) Here is a bullet point list of suspicious behavior (if any) to highlight.
"

# #########################################
# ### Merge Chunk Summaries + Anomaly Score
# #########################################

# merge the chunk summaries and anomaly score
summaries_file=$(basename "$input_video" .mp4).json
python summary_merger.py "$output_dir/$summaries_file" -d "GPU" -m "$llm_model"
