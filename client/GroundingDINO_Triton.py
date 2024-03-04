import numpy as np
import tritonclient.grpc as grpcclient
import torch

from groundingdino.util import get_tokenlizer
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map



class GroundingDINO:
    def __init__(self, url):
        self.client = grpcclient.InferenceServerClient(url=url)

    def get_backbone_embeds(self, img) -> list:
        """Backbone inference"""
        img = img.unsqueeze(0)

        detection_input = grpcclient.InferInput("images", img.shape, datatype="FP32")
        detection_input.set_data_from_numpy(img.numpy())

        detection_response = self.client.infer(model_name="DINO_backbone", inputs=[detection_input])
        srcs = [detection_response.as_numpy("9590"),
                detection_response.as_numpy("9591"),
                detection_response.as_numpy("9592"),
                detection_response.as_numpy("9593")]
        return srcs


    def get_text_embeds(self, tokens)-> np.ndarray:
        """Bert inference"""
        position_ids = grpcclient.InferInput("position_ids", tokens['position_ids'].shape, datatype="INT64")
        position_ids.set_data_from_numpy(tokens['position_ids'].numpy())
        
        attention_mask = grpcclient.InferInput("attention_mask", tokens['attention_mask'].shape, datatype="BOOL")
        attention_mask.set_data_from_numpy(tokens['attention_mask'].numpy())

        input_ids = grpcclient.InferInput("input_ids", tokens['input_ids'].shape, datatype="INT64")
        input_ids.set_data_from_numpy(tokens['input_ids'].numpy())
        
        encoded_text = self.client.infer(model_name="DINO_bert", inputs=[position_ids, attention_mask, input_ids])
        
        return encoded_text.as_numpy("encoded_text")

    def transformer_inference(self, text_embeds, tokens, srcs):
        src0 = grpcclient.InferInput("src0", srcs[0].shape, datatype="FP32")
        src0.set_data_from_numpy(srcs[0])
        
        src1 = grpcclient.InferInput("src1", srcs[1].shape, datatype="FP32")
        src1.set_data_from_numpy(srcs[1])
        
        src2 = grpcclient.InferInput("src2", srcs[2].shape, datatype="FP32")
        src2.set_data_from_numpy(srcs[2])
        
        src3 = grpcclient.InferInput("src3", srcs[3].shape, datatype="FP32")
        src3.set_data_from_numpy(srcs[3])
        
        position_ids = grpcclient.InferInput("position_ids", tokens['position_ids'].shape, datatype="INT64")
        position_ids.set_data_from_numpy(tokens['position_ids'].numpy())
        
        text_self_attention_masks = grpcclient.InferInput("text_self_attention_masks", tokens['attention_mask'].shape, datatype="BOOL")
        text_self_attention_masks.set_data_from_numpy(tokens['attention_mask'].numpy())
        
        encoded_text = grpcclient.InferInput("encoded_text", text_embeds.shape, datatype="FP32")
        encoded_text.set_data_from_numpy(text_embeds)
        
        inference_result = self.client.infer(model_name="DINO_transformer", inputs=[src0,
                                                                            src1,
                                                                            src2,
                                                                            src3,
                                                                            position_ids,
                                                                            text_self_attention_masks,
                                                                            encoded_text])
        
        return inference_result.as_numpy("outputs_coord_list"), inference_result.as_numpy("outputs_class")
        
        
    def inference(self, text_prompt, box_treshold, text_treshold, image):

        srcs = self.get_backbone_embeds(image)
        tokenizer = get_tokenlizer.get_tokenlizer("bert-base-uncased")
        specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        tokenized = tokenizer(text_prompt, padding="longest", return_tensors="pt")

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(tokenized, specical_tokens, tokenizer)

        tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks
        tokenized_for_encoder["position_ids"] = position_ids

        text_embeds = self.get_text_embeds(tokenized_for_encoder)


        prediction_boxes, prediction_logits= self.transformer_inference(text_embeds, tokenized_for_encoder, srcs)

        prediction_logits = prediction_logits[0][0]
        prediction_boxes = prediction_boxes[0][0]
        mask = np.max(prediction_logits,axis=1) > box_treshold
        # print(mask)
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
        print(logits.shape)
        logits = logits.max(axis=1)
        return boxes, logits