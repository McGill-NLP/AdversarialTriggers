#!/bin/bash

if [ -z "${SCRATCH}" ]; then
    # If SCRATCH is not defined, we are likely launching a DGX job from a local
    # machine.
    export SCRATCH="/scratch"
fi

PROJECT_DIR="${SCRATCH}/AdversarialTriggers"

job_script () {
    # Infer what cluster we are on and return the job script.
    local server=$(hostname -f)
    if [[ "${server}" == *"login-"* ]]; then
        cluster="mila"
    elif [[ "${server}" == *"narval"* ]]; then
        cluster="narval"
    else
        cluster="dgx"
    fi

    local jobscript="python_${cluster}_job.sh"

    if [ ! -f "${jobscript}" ]; then
        echo "Job script ${jobscript} not found."
        return 1
    fi

    echo "${jobscript}"
}

join_by_with_prefix () {
    # Joins a string with a prefix.
    string=$1
    prefix=$2

    IFS=' ' read -r -a array <<< "${string}"

    for i in "${!array[@]}"; do
        array[i]="${prefix}/${array[i]}"
    done

    echo "${array[@]}"
}

submit_seed () {
    # Submits a SLURM job or NGC job to the cluster.
    local walltime=$1
    local jobscript=$2
    local experiment_id

    # Try to create a experiment ID.
    if ! experiment_id=$(python3 adversarial_triggers/experiment_utils.py "${@:4}"); then
        echo "Error creating experiment ID."
        return 1
    fi

    if [[ "${walltime}" == *"?"* ]]; then
        echo "Undefined walltime for ${experiment_id}."
        return 0
    fi

    if [[ "${jobscript}" == "python_dgx_job.sh" ]]; then
        commandline="source setup.sh && bash ${@:2}"

        ngc base-command job run \
            --name "${experiment_id}" \
            --priority "NORMAL" \
            --order "50" \
            --preempt "RUNONCE" \
            --min-timeslice "0s" \
            --total-runtime "0s" \
            --ace "${ACE}" \
            --org "${ORG}" \
            --instance "${INSTANCE}" \
            --result "${RESULT}" \
            --image "${IMAGE}" \
            --workspace "${WORKSPACE}" \
            --env-var "experiment_id:${experiment_id}" \
            --label "q_$(date +'%Y_%m_%d')" \
            --port "8383" \
            --commandline "${commandline}"

        return 0
    fi

    # Extract the root experiment name.
    experiment_name=${experiment_id%%_*}

    # Check if results already exist.
    if [ -f "${PROJECT_DIR}/results/${experiment_name}/${experiment_id}.json" ]; then
        echo "Experiment ${experiment_id} already completed."
        return 0
    fi

    # Try to submit job.
    if job_id=$(
        sbatch \
            --parsable \
            --time "${walltime}" \
            -J "${experiment_id}" \
            -o "${PROJECT_DIR}/logs/%x.%j.out" \
            -e "${PROJECT_DIR}/logs/%x.%j.err" \
            "${@:2}"
    ); then
        echo "Submitted job ${job_id}."
    else
        echo "Error submitting job ${job_id}."
        return 1
    fi
}
