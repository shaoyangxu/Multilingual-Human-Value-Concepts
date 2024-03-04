from transformers.pipelines import TextGenerationPipeline
from .rep_control_reading_vec import WrappedReadingVecModel

class RepControlPipeline(TextGenerationPipeline):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 layers, 
                 block_name="decoder_block", 
                 control_method="reading_vec",
                 **kwargs):
        
        # TODO: implement different control method and supported intermediate modules for different models
        assert control_method == "reading_vec", f"{control_method} not supported yet"
        assert block_name == "decoder_block" or "LlamaForCausalLM" in model.config.architectures, f"{model.config.architectures} {block_name} not supported yet"
        self.wrapped_model = WrappedReadingVecModel(model, tokenizer)
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(layers, block_name=block_name)
        self.block_name = block_name
        self.layers = layers
        """
        wrapped_block:
            reset
            set_controller(
                activations->controller, token_pos=None, masks=None, normalize=False, operator='linear_comb'
            )->self.operator = xx
                operater: paper p 13
                    linear_comb: def op(current, controller): return current + controller
                    piecewise_linear: sign = torch.sign((current * controller).sum(-1, keepdim=True))
                        p 26: The piece-wise operator achieves the best helpful and harmless rates in these settings.
            forward-control
                # norm_pre = torch.norm(modified, dim=-1, keepdim=True)
                token-wise
        """
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
   
    def __call__(self, text_inputs, activations=None, **kwargs):

        if activations is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name)
        """
        文本输入，
        """
        outputs = super().__call__(text_inputs, **kwargs)
        self.wrapped_model.reset()

        return outputs