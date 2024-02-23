from transformers import PretrainedConfig

class BEFREConfig(PretrainedConfig):

    def __init__(
        self,
        pretrained_model_name_or_path=None,
        cache_dir=None,
        revision="main",
        use_auth_token=False,
        hidden_dropout_prob=0.1,
        max_span_width=30,
        use_span_width_embedding=False,
        linear_size=128,
        init_temperature=0.07,
    ):
        self.pretrained_model_name_or_path=pretrained_model_name_or_path
        self.cache_dir=cache_dir
        self.revision=revision
        self.use_auth_token=use_auth_token
        self.hidden_dropout_prob=hidden_dropout_prob
        self.max_span_width = max_span_width
        self.use_span_width_embedding = use_span_width_embedding
        self.linear_size = linear_size
        self.init_temperature = init_temperature