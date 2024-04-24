class _AnnotationMapping(dict):
    """Annotations for plotting and creating tables.

    Maps internal names onto pretty names for plotting and table creation.
    """

    def labeller(self, key):
        return self.get(key, key)

    def breaks(self):
        return list(self.keys())

    def labels(self):
        return list(self.values())


class _AllAnnotations:
    def __init__(self):
        self.model_family = _AnnotationMapping(
            {
                "gemma-1.1-2b-it": "Gemma",
                "gemma-1.1-7b-it": "Gemma",
                "guanaco-7B-HF": "Vicuna",
                "guanaco-13B-HF": "Vicuna",
                "koala-7B-HF": "Koala",
                "Llama-2-7b-chat-hf": "Llama2",
                "Llama-2-13b-chat-hf": "Llama2",
                "mpt-7b-chat": "MPT",
                "openchat_3.5": "OpenChat",
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-dromedary_l-2e-05_s-0": "Llama2",
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-lima_l-2e-05_s-0": "Llama2",
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-saferpaca2000_l-2e-05_s-0": "Llama2",
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-sharegpt_l-2e-05_s-0": "Llama2",
                "Starling-LM-7B-alpha": "OpenChat",
                "Starling-LM-7B-beta": "OpenChat",
                "vicuna-7b-v1.5": "Vicuna",
                "vicuna-13b-v1.5": "Vicuna",
            }
        )

        self.model = _AnnotationMapping(
            {
                "gemma-1.1-2b-it": "Gemma-2B-Chat",
                "gemma-1.1-7b-it": "Gemma-7B-Chat",
                "guanaco-7B-HF": "Guanaco-7B",
                "guanaco-13B-HF": "Guanaco-13B",
                "koala-7B-HF": "Koala-7B",
                "Llama-2-7b-chat-hf": "Llama2-7B-Chat",
                "Llama-2-13b-chat-hf": "Llama2-13B-Chat",
                "mpt-7b-chat": "MPT-7B-Chat",
                "openchat_3.5": "OpenChat-3.5-7B",
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-dromedary_l-2e-05_s-0": "SelfAlign-7B",
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-lima_l-2e-05_s-0": "Lima-7B",
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-saferpaca2000_l-2e-05_s-0": "Saferpaca-7B",
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-sharegpt_l-2e-05_s-0": "DistilLlama2-7B",
                "Starling-LM-7B-alpha": "Starling-7B-α",
                "Starling-LM-7B-beta": "Starling-7B-β",
                "vicuna-7b-v1.5": "Vicuna-7B",
                "vicuna-13b-v1.5": "Vicuna-13B",
            }
        )

        self.model_ordinal = _AnnotationMapping(
            {
                "gemma-1.1-2b-it": 0,
                "gemma-1.1-7b-it": 1,
                "guanaco-7B-HF": 2,
                "guanaco-13B-HF": 3,
                "koala-7B-HF": 4,
                "Llama-2-7b-chat-hf": 5,
                "Llama-2-13b-chat-hf": 6,
                "mpt-7b-chat": 7,
                "openchat_3.5": 8,
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-dromedary_l-2e-05_s-0": 9,
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-lima_l-2e-05_s-0": 10,
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-saferpaca2000_l-2e-05_s-0": 11,
                "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-sharegpt_l-2e-05_s-0": 12,
                "Starling-LM-7B-alpha": 13,
                "Starling-LM-7B-beta": 14,
                "vicuna-7b-v1.5": 15,
                "vicuna-13b-v1.5": 16,
            }
        )

        self.dataset = _AnnotationMapping(
            {
                "behaviour": "AdvBench (Seen)",
                "unseen": "AdvBench (Unseen)",
                "controversial": "I-Controversial",
                "cona": "I-CoNa",
                "malicious": "MaliciousInstruct",
                "qharm": "Q-Harm",
                "sharegpt": "ShareGPT",
                "string": "String",
            }
        )

        self.dataset_ordinal = _AnnotationMapping(
            {
                "behaviour": 0,
                "unseen": 1,
                "controversial": 2,
                "cona": 3,
                "malicious": 4,
                "qharm": 5,
                "sharegpt": 6,
                "string": 7,
            }
        )


annotation = _AllAnnotations()
