
from abc import ABC
import logging

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class MotionGeneratorHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(MotionGeneratorHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-small")

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        self.initialized = True

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes. 
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)

        inputs = self.tokenizer(
            sentences, 
            max_length=512, 
            padding="max_length", 
            truncation=True, 
            return_attention_mask=True, 
            add_special_tokens=True, 
            return_tensors="pt"
        )

        return inputs

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        # NOTE: This makes the assumption that your model expects text to be tokenized  
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit 
        # its expected input format.

        generated_ids = self.model.generate(input_ids=inputs["input_ids"].to(self.device),
                    attention_mask=inputs["attention_mask"].to(self.device),
                    max_length=64,
                    min_length = 2,
                    temperature=1.2,
                    num_beams=5,
                    repetition_penalty=32.0,
                    length_penalty=8.0,
                    early_stopping=True,
                    use_cache=False,
                    top_k=0,
                    do_sample=True, 
                    num_return_sequences=5
                    )

        prediction = [
            self.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
        ]

        logger.info("Model predicted: '%s'", prediction)


        return [prediction]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = MotionGeneratorHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e