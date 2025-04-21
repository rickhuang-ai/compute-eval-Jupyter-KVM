#!usr/bin/bash
ans_PATH="/home/ailab/files/compute-eval-Jupyter-KVM"

# Calculate total valid results
total_valid_results=$(cat ${ans_PATH}/ans_samples.${1}_correctness_results.jsonl | grep -v '"result": "Failed to run!' | wc -l)
echo "total_valid_results = ${total_valid_results}"

# Calculate passed results
passed_results=$(cat ${ans_PATH}/ans_samples.${1}_correctness_results.jsonl | grep '"result": "passed"' | wc -l)
echo "passed_results = ${passed_results}"

# Calculate code generation accuracy
if [ "$total_valid_results" -ne 0 ]; then
    code_gen_accuracy=$(echo "scale=2; $passed_results / $total_valid_results" | bc)
else
    code_gen_accuracy=0
fi
echo "Code generation tasks average accuracy = ${code_gen_accuracy}"


# total_valid_results=0
# cat ${ans_PATH}/ans_samples.${1}_correctness_results.jsonl | grep -v '"result": "Failed to run!' | wc -l > ${total_valid_results}
# # save above result to a bash var
# echo "total_valid_results = ", ${total_valid_results}
# # [TODO] paste history command to get passed_results from jsonl
# passed_result=0
# cat ${ans_PATH}/ans_samples.${1}_correctness_results.jsonl | egrep '*"result": "passed",*' | wc -l > ${passed_results}
# echo "passed_results = ", ${passed_results}

# code_gen_accuracy=0
# # Calculate total valid results
# total_valid_results=$(cat ${ans_PATH}/ans_samples.${1}_correctness_results.jsonl | grep -v '"result": "Failed to run!' | wc -l)
# echo "total_valid_results = ${total_valid_results}"
