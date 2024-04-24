MODEL_NAME_OR_PATH = \
	google/gemma-1.1-2b-it \
	google/gemma-1.1-7b-it \
	TheBloke/guanaco-7B-HF \
	TheBloke/guanaco-13B-HF \
	TheBloke/koala-7B-HF \
	meta-llama/Llama-2-7b-chat-hf \
	meta-llama/Llama-2-13b-chat-hf \
	mosaicml/mpt-7b-chat \
	openchat/openchat_3.5 \
	berkeley-nest/Starling-LM-7B-alpha \
	Nexusflow/Starling-LM-7B-beta \
	lmsys/vicuna-7b-v1.5 \
	lmsys/vicuna-13b-v1.5

sync-mila:	
	rsync --progress -urltv --delete \
		--filter=":- .gitignore" \
		-e ssh . mila:~/workspace/AdversarialTriggers

sync-narval:
	rsync --progress -urltv --delete \
		--filter=":- .gitignore" \
		-e ssh . narval:~/workspace/AdversarialTriggers

download-results-mila:
	rsync --progress -urltv --compress \
		-e ssh mila:~/scratch/AdversarialTriggers/results/ ./results

download-results-narval:
	rsync --progress -urltv --compress \
		-e ssh narval:~/scratch/AdversarialTriggers/results/ ./results

download-tensorboard-mila:
	rsync --progress -urltv --compress \
		-e ssh mila:~/scratch/AdversarialTriggers/tensorboard/ ./tensorboard

download-tensorboard-narval:
	rsync --progress -urltv --compress \
		-e ssh narval:~/scratch/AdversarialTriggers/tensorboard/ ./tensorboard

paper-tables:
	python3 export/system_message_table.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

	python3 export/chat_template_table.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

	python3 export/single_asr_table.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

paper-plots:	
	python3 export/main_advbench_seen_partial.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

	python3 export/main_advbench_seen_all.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

	python3 export/safety_advbench_seen_partial.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

	python3 export/safety_advbench_seen_all.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

	python3 export/AFT_train_partial.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

	python3 export/AFT_train_all.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)
	
	python3 export/AFT_absolute_partial.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

	python3 export/AFT_transfer_partial.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

	python3 export/instruction_generalization_partial.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

	python3 export/instruction_generalization_all.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH)

	python3 export/all_dataset.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH) \
		--dataset_name "behaviour"

	python3 export/all_dataset.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH) \
		--dataset_name "behaviour"

	python3 export/all_dataset.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH) \
		--dataset_name "unseen"

	python3 export/all_dataset.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH) \
		--dataset_name "cona"

	python3 export/all_dataset.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH) \
		--dataset_name "controversial"

	python3 export/all_dataset.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH) \
		--dataset_name "malicious"

	python3 export/all_dataset.py \
		--model_name_or_path $(MODEL_NAME_OR_PATH) \
		--dataset_name "qharm"

clean:
	rm -rf ./results
	rm -rf ./plot
	rm -rf ./table
	rm -rf ./tensorboard
