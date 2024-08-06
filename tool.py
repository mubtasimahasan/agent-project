from transformers import Blip2Processor, Blip2ForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
import torch
import spacy
import easyocr
from PIL import Image
from difflib import SequenceMatcher
import requests
import torchvision.models as models
import torchvision.transforms as T
from bs4 import BeautifulSoup
from collections import OrderedDict
import json
import numpy as np

# Tools
class ImageCaptionTool:
    def __init__(self, blip2_model, blip2_processor):
        self.processor = blip2_processor
        self.model = blip2_model
        
    def execute(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        out = self.model.generate(**inputs, max_new_tokens=100)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

class ObjectDetectionTool:
    def __init__(self, detr_model, detr_processor):
        self.processor = detr_processor
        self.model = detr_model
        self.nlp = spacy.load("en_core_web_sm")

    def execute(self, query, image):
        if np.array(image).ndim > 3:
            image = image[..., :3]
            image = Image.fromarray(image)

        elif np.array(image).ndim != 3:
            image = np.stack([image]*3, axis=-1)
            image = Image.fromarray(image)

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detected_boxes = []
        detected_labels = []

        query_doc = self.nlp(query.lower())
        query_keywords = [token.text for token in query_doc if not token.is_stop]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model.config.id2label[label.item()].lower()
            for keyword in query_keywords:
                keyword_doc = self.nlp(keyword)
                label_doc = self.nlp(label_name)
                similarity = keyword_doc.similarity(label_doc)
                if similarity > 0.1:
                    detected_boxes.append(box)
                    detected_labels.append(label_name)
                    break

        if not detected_labels:
            return "No matching objects found for the query."

        object_counts = {}
        for label in detected_labels:
            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1

        detection_summary = {
            "detected_objects": detected_labels,
            "object_counts": object_counts,
        }

        return detection_summary

class RegionGroundTool:
    def __init__(self, detr_model, detr_processor):
        self.processor = detr_processor
        self.model = detr_model
        self.nlp = spacy.load("en_core_web_sm")

    def execute(self, query, image):
        if np.array(image).ndim > 3:
            image = image[..., :3]
            image = Image.fromarray(image)

        elif np.array(image).ndim != 3:
            image = np.stack([image]*3, axis=-1)
            image = Image.fromarray(image)

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detected_boxes = []
        detected_labels = []
        confidence_scores = []

        query_doc = self.nlp(query.lower())

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_name = self.model.config.id2label[label.item()].lower()
            label_doc = self.nlp(label_name)
            similarity = query_doc.similarity(label_doc)
            if similarity > 0.1:
                detected_boxes.append(box)
                detected_labels.append(label_name)
                confidence_scores.append(score.item())

        if not detected_boxes:
            print("No matching regions found for the query.")
            return None

        max_confidence_index = confidence_scores.index(max(confidence_scores))
        most_confident_box = detected_boxes[max_confidence_index]
        cropped_image = image.crop(most_confident_box)

        return cropped_image

class TextDetectionTool:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu="cuda:0")

    def execute(self, image):
        results = self.reader.readtext(image)
        detected_text = " ".join([result[1] for result in results])
        
        if not results:
            return 'No text detected'
            
        return detected_text

class TextGroundTool:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu="cuda:0")

    def polygon_to_bbox(self, polygon):
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    def calculate_similarity(self, text1, text2):
        return SequenceMatcher(None, text1, text2).ratio()

    def execute(self, query, image):
        detection_results = self.reader.readtext(image)
        matched_boxes = []
        for result in detection_results:
            box, detected_text, _ = result
            detected_text = detected_text.lower()
            similarity = self.calculate_similarity(detected_text, query.lower())
            if similarity > 0.5:
                matched_boxes.append(self.polygon_to_bbox(box))

        if not matched_boxes:
            return "No matching text found for the query."

        left, top, right, bottom = matched_boxes[0]
        cropped_image = image.crop((left, top, right, bottom))

        return cropped_image

class ImageClassificationTool:
    def __init__(self):
        self.model = models.resnet50(pretrained=True).eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.labels = self._load_labels()

    def _load_labels(self):
        labels_path = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        labels = requests.get(labels_path).json()
        return labels

    def execute(self, image):
        if np.array(image).ndim > 3:
            image = image[..., :3]
            image = Image.fromarray(image)

        elif np.array(image).ndim != 3:
            image = np.stack([image]*3, axis=-1)
            image = Image.fromarray(image)

        image = Image.fromarray(np.array(image))
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image)
        _, predicted = torch.max(outputs, 1)
        return self.labels[predicted[0].item()]

class ContextReasoningTool:
    def __init__(self, blip2_model, blip2_processor):
        self.processor = blip2_processor
        self.model = blip2_model
        
    def execute(self, question, image):
        prompt = f'''
        You are an expert reasoning tool designed to answer questions by leveraging commonsense, context, and internal reasoning in conjunction with analyzing an input image and question.
        
        Your task:
        - Carefully analyze the question and the provided image.
        - Apply relevant commonsense and contextual reasoning to derive an answer.
        - Your response should be concise, relevant, and derived from a combination of image analysis and internal reasoning.

        Example:
        Question: What is the capital city of the country represented by the flag in the image?
        Image Description: A flag with red, white, and blue stripes and a star in the upper left corner.
        Internal Knowledge: The flag described matches the United States flag, whose capital city is Washington, D.C.
        Answer: Washington, D.C.

        Current Task:
        Question: {question}
        Answer:
        '''
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        return answer

class KnowledgeReasoningTool:
    def __init__(self, blip2_model, blip2_processor, detr_model, detr_processor):
        self.vlm_processor = blip2_processor
        self.vlm_model = blip2_model
        self.od_processor = detr_processor
        self.od_model = detr_model

    def format_object_name(self, object_name):
        name_parts = object_name.split(' ')
        name_parts[0] = name_parts[0].capitalize()
        if len(name_parts) > 1:
            name_parts[1:] = [part.lower() for part in name_parts[1:]]
        formatted_object_name = '_'.join(name_parts)
        return formatted_object_name

    def detect_objects(self, image, threshold=0.7):
        inputs = self.od_processor(images=image, return_tensors="pt")
        outputs = self.od_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.od_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
        object_names = list(set([self.format_object_name(self.od_model.config.id2label[label.item()]) for label in results["labels"]]))
        return object_names

    def get_wiki_summary(self, object_name, retries=3, timeout=5):
        url = f"https://en.wikipedia.org/wiki/{object_name}"
        headers = {'User-Agent': 'YourCustomUserAgent/1.0 (yourname@example.com)'}
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    summary_paragraphs = soup.find_all('p', limit=3)
                    summary = " ".join([para.text for para in summary_paragraphs]).replace('\n', '')
                    return summary
                else:
                    print(f"Request to {url} returned status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
        return None

    def extract_external_knowledge(self, object_names, kb_type):
        external_knowledge = None
        all_object_details = []
        if kb_type == 'unstructured':
            for object_name in object_names:
                wiki_summary = self.get_wiki_summary(object_name)
                if wiki_summary is not None:
                    object_details = OrderedDict()
                    object_details['name'] = object_name
                    object_details['information'] = wiki_summary
                    all_object_details.append(dict(object_details))
        elif kb_type == 'structured':
            pass
        if len(all_object_details) != 0:
            external_knowledge = json.dumps(all_object_details, indent=2)
        return external_knowledge

    def execute(self, question, image):
        if np.array(image).ndim > 3:
            image = image[..., :3]
            image = Image.fromarray(image)

        elif np.array(image).ndim != 3:
            image = np.stack([image]*3, axis=-1)
            image = Image.fromarray(image)
            
        detected_object_names = self.detect_objects(image)
        external_unstructured_knowledge = self.extract_external_knowledge(detected_object_names, kb_type='unstructured')
        external_knowledge = external_unstructured_knowledge
        example_external_knowledge = [{
            'name': 'flag',
            'information': 'A flag is a piece of fabric (most often rectangular) with distinctive colours and design. It is used as a symbol, a signalling device, or for decoration. The term flag is also used to refer to the graphic design employed, and flags have evolved into a general tool for rudimentary signalling and identification, especially in environments where communication is challenging (such as the maritime environment, where semaphore is used). Many flags fall into groups of similar designs called flag families.[1] The study of flags is known as \"vexillology\" from the Latin vexillum, meaning \"flag\" or \"banner\".'
        }]
        example_external_knowledge = json.dumps(example_external_knowledge, indent=2)
        prompt = f"""
        You are an expert reasoning tool designed to answer questions by leveraging external knowledge and analyzing an input image and question. Your task is to utilize both the visual information from the image and information provided by external knowledge to predict accurate answers.

        Your task:
        - Carefully analyze the question and the provided image.
        - Utilize the information provided by external knowledge to derive an answer.
        - Your response should be concise, relevant, and derived from a combination of image analysis and internal reasoning.

        Example:
        Question: What is the capital city of the country represented by the flag in the image?
        Image: A flag with red, white, and blue stripes and a star in the upper left corner.
        External Knowledge: {example_external_knowledge}
        Answer: Washington, D.C.

        Current Task:
        Question: {question}
        External Knowledge: {external_knowledge}
        New Answer:
        """
        inputs = self.vlm_processor(images=image, text=prompt, return_tensors="pt")
        outputs = self.vlm_model.generate(**inputs, max_new_tokens=50)
        generated_text = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split("New Answer:")[-1].strip()
        return answer

# Instantiate tools
def initialize_tools():
    blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
    
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    caption_tool = ImageCaptionTool(blip2_model, blip2_processor)
    object_detection_tool = ObjectDetectionTool(detr_model, detr_processor)
    region_ground_tool = RegionGroundTool(detr_model, detr_processor)
    text_detection_tool = TextDetectionTool()
    text_ground_tool = TextGroundTool()
    image_classification_tool = ImageClassificationTool()
    context_reasoning_tool = ContextReasoningTool(blip2_model, blip2_processor)
    knowledge_reasoning_tool = KnowledgeReasoningTool(blip2_model, blip2_processor, detr_model, detr_processor)

    return caption_tool, object_detection_tool, region_ground_tool, text_detection_tool, text_ground_tool, image_classification_tool, context_reasoning_tool, knowledge_reasoning_tool
    

    