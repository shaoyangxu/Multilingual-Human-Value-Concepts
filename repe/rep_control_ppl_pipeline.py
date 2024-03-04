from transformers import Pipeline
from typing import List, Union, Optional
from .rep_control_reading_vec import WrappedReadingVecModel
import torch

class RepControlPplPipeline(Pipeline):
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
   
    
    def _sanitize_parameters(self, 
                             **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        forward_params =  {}
        postprocess_params = {}
        return preprocess_params, forward_params, postprocess_params
    
    def preprocess(
            self, 
            inputs: Union[str, List[str], List[List[str]]],
            **tokenizer_kwargs):
        if self.image_processor:
            return self.image_processor(inputs, add_end_of_utterance_token=False, return_tensors="pt")
        return self.tokenizer(inputs, return_tensors=self.framework, **tokenizer_kwargs)

    def postprocess(self, outputs):
        return outputs

    def _forward(self, model_inputs):
        """
        Args:
        - which_hidden_states (str): Specifies which part of the model (encoder, decoder, or both) to compute the hidden states from. 
                        It's applicable only for encoder-decoder models. Valid values: 'encoder', 'decoder'.
        """
        # get model hidden states and optionally transform them with a RepReader
        bs = model_inputs['input_ids'].shape[0]
        attention_mask = model_inputs["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        model_inputs["position_ids"] = position_ids
        with torch.no_grad():
            outputs =  self.model(**model_inputs)
        outputs['input_ids'] = model_inputs['input_ids']
        return outputs
    
    def __call__(self, text_inputs, activations=None, **kwargs):

        if activations is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name)
        outputs = super().__call__(text_inputs, **kwargs)
        self.wrapped_model.reset()
        return outputs