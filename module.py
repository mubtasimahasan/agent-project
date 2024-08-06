from utility import map_text_to_choices

from PIL import Image
import re 

class Planner:
    def __init__(self, tools, model, processor):
        self.tools = tools
        self.model = model
        self.processor = processor

    def plan(self, question, image, retry_count=10):
        plan_prompt = f'''
        USER: You are a planning agent designed to answer questions based on images by selecting the appropriate tools from a given list. Your task is to analyze the input question and image, determine which tools are necessary, and sequence these tools to extract the required information to answer the question.

        Below is a list of available tools and their functionalities:

        ImageCaptionTool - Generates captions describing the image content. Select this tool when the question requires understanding the overall scene and context of the image.
        ObjectDetectionTool - Identifies and counts objects mentioned in the query. Select this tool when the question requires verifying the presence and quantity of specific objects in the image.
        RegionGroundTool - Locates specific areas of the image based on the query. Select this tool when the question requires focusing on and analyzing relevant regions within the image.
        TextDetectionTool - Extracts text embedded within the image (OCR). Select this tool when the question requires understanding and interpreting textual information present in the image.
        TextGroundTool - Finds specific text locations in the image. Select this tool when the question requires identifying precise areas where specific text is located for further analysis.
        ImageClassificationTool - Categorizes the image into predefined classes. Select this tool when the question requires classifying objects or scenes within the image.
        ContextReasoningTool - Infers answers based on commonsense, reasoning, and the context provided by the image. Select this tool when the question requires drawing inferences, making predictions, and applying reasoning. 
        KnowledgeReasoningTool - Infers answers based on external knowledge and information retrieval. Select this tool when the question requires additional information to supplement the analysis and provide accurate answers beyond basic image understanding.
        
        Your task:
        - Select the tools required to answer the question by processing the input question and image. See the examples below to understand which tools are selected and the rationale behind their selection.
        - Your response must be in a single line, and you must only write the tool names separated by commas.
        - Your response must not contain any numbers, bullet points, or special symbols, and do not include any explanations or additional text in your answer. 
        - Your response must be in the exact format specified below. Do not deviate from this format.

        Example 1:
        Input: An image of a teddy bear cake on a table.
        Question: Which number birthday is probably being celebrated?
        Explanation: Here, the planner will select TextDetectionTool to extract and read the text present on the cake in the image. The ContextReasoningTool is then selected to apply context and infer the meaning or relevance of the extracted text.
        ASSISTANT: TextDetectionTool, ContextReasoningTool

        Example 2:
        Input: An image of a group of giraffes standing under a tree near a pond.
        Question: What best describes the pool of water?
        Explanation: Here, the planner will select ImageCaptionTool to generate an overall description of the scene, which includes the number of giraffes. Additionally, the ObjectDetectionTool is selected to accurately identify and count the giraffes in the image. The KnowledgeReasoningTool is then used to provide additional details.
        ASSISTANT: ImageCaptionTool, ObjectDetectionTool, KnowledgeReasoningTool

        Current Task:
        Input: <image>
        Question: {question}
        Your response format - tool name, tool name,
        ASSISTANT:
        '''
        inputs = self.processor(plan_prompt, image, padding=True, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.9)
        generated_text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        # print("Full interaction with planner:", generated_text)
        generated_text = generated_text.split("ASSISTANT:")[-1].strip()
        # print("Planner Reply:", generated_text)
        selected_tools = []
        for tool_name in self.tools:
            if tool_name in generated_text:
                selected_tools.append(tool_name) 
        if not selected_tools:
            if retry_count > 0:
                print(f"Planner failed to select any tools. Retrying... Attempts left: {retry_count}")
                return self.plan(question, image, retry_count - 1)
            else:
                print("Planner failed to select any tools after maximum retries. Selecting all tools.")
                selected_tools = list(self.tools.keys())
        
        print(f"{'~' * 26} Selected Tools by Planner: {'~' * 25} \n {selected_tools}")
        return selected_tools

class Executor:
    def __init__(self, tools):
        self.tools = tools

    def execute(self, question, image, selected_tools):
        results = {}
        for tool_name in selected_tools:
            tool = self.tools[tool_name]
            
            if tool_name in ("ObjectDetectionTool", "ContextReasoningTool", "KnowledgeReasoningTool"):
                results[tool_name] = tool.execute(question, image)
            elif tool_name == "RegionGroundTool":
                region_result = tool.execute(question, image)
                if isinstance(region_result, Image.Image):
                    results["RegionGroundTool->ImageCaptionTool"] = self.tools["ImageCaptionTool"].execute(region_result)
                else:
                    results["RegionGroundTool"] = region_result
            elif tool_name == "TextGroundTool":
                text_ground_result = tool.execute(question, image)
                if isinstance(text_ground_result, Image.Image):
                    results["TextGroundTool->TextDetectionTool"] = self.tools["TextDetectionTool"].execute(text_ground_result)
                else:
                    results["TextGroundTool"] = text_ground_result
            else:
                results[tool_name] = tool.execute(image)

            if tool_name not in self.tools:
                print(f"Warning: Unknown tool name: {tool_name}")
        
        print(f"{'=' * 33} Tools Output: {'=' * 31} \n {results}")
        return results

class Answerer:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def multiple_choice_answer(self, question, results, choices):
        answer_prompt = f"""You are an advanced visual question-answering tool. Choose the correct answer from the choices based on the tool outputs.  If the tool outputs do not provide enough information, select an answer randomly from the choices.
        
        Example 1:
        <s>[INST]
        Previous Tool Outputs: {{'TextDetectionTool': 'J,2 [H] 100-0237', 'ContextReasoningTool': 'frosting'}}
        Question: What is the topping on the cake?
        Choices: ['butter', 'mayo', 'ice cream', 'icing']
        Answer: icing
        [/INST]</s>
        
        Example 2:
        <s>[INST]
        Previous Tool Outputs: {{'ObjectDetectionTool': {{'detected_objects': ['person', 'cell phone', 'person', 'person', 'motorcycle', 'car'], 'object_counts': {{'person': 3, 'cell phone': 1, 'motorcycle': 1, 'car': 1}}, 'TextDetectionTool': 'No text detected', 'ContextReasoningTool': 'a cigar'}}
        Question: What is in the motorcyclist's mouth?
        Choices: ['toothpick', 'food', 'popsicle stick', 'cigarette']
        Answer: cigarette
        [/INST]</s>
        
        Current task:
        [INST]
        Previous Tool Outputs: {results}
        Question: {question}
        Choices: {choices}
        Answer: ?
        [/INST]
        """
        
        inputs = self.tokenizer(answer_prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id)
        decoded_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer_text = decoded_text.split("Answer: ?")[-1].split("[/INST]")[-1].strip()
        # print("Answerer Reply:", answer_text)
        answer = map_text_to_choices(answer_text, choices)
        #print(f"{'+' * 22} Predicted Multiple Choice Answer: {'+' * 22} \n {answer}")
        return answer
    
    def direct_answer(self, question, results):
        answer_prompt = f"""You are an advanced visual question-answering tool. 
        
        Your task:
        - Provide a concise answer (one or two words) based on the tool outputs. 
        - If the tool outputs do not provide enough information, guess a probable answer.
        - You must give only the answer and do not include additional context or explanation.

        Example 1:
        <s>[INST]
        Previous Tool Outputs: {{'TextDetectionTool': 'J,2 [H] 100-0237', 'ContextReasoningTool': 'frosting'}}
        Question: What is the topping on the cake?
        Answer: icing
        [/INST]</s>
        
        Example 2:
        <s>[INST]
        Previous Tool Outputs: {{'ObjectDetectionTool': {{'detected_objects': ['person', 'cell phone', 'person', 'person', 'motorcycle', 'car'], 'object_counts': {{'person': 3, 'cell phone': 1, 'motorcycle': 1, 'car': 1}}, 'TextDetectionTool': 'No text detected', 'ContextReasoningTool': 'a cigar'}}
        Question: What is in the motorcyclist's mouth?
        Answer: cigarette
        [/INST]</s>
        
        Current task:
        [INST]
        Previous Tool Outputs: {results}
        Question: {question}
        Answer (one or two words): 
        [/INST]
        """
        
        inputs = self.tokenizer(answer_prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=10, 
                                     pad_token_id=self.tokenizer.eos_token_id,
                                     do_sample=True,
                                     temperature=0.5,
                                     top_k=10)
        decoded_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded_text.split("Answer (one or two words):")[-1].split("[/INST]")[-1].strip()
        #print(f"{'^' * 26} Predicted Direct Answer: {'^' * 27} \n {answer}")
        return answer
    
class Evaluator:
    def __init__(self, tokenizer, model):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, question, tool_outputs, mc_answer, da_answer):
        eval_prompt = f'''
        USER:
        You are an evaluation agent expert in analyzing various tools' outputs, the predicted multiple choice answer, and the direct answer, evaluating their usefulness in the context of the question, and providing a decision and score to guide the planning agent. 
        
        Your tasks:
        1. Evaluate whether the output from the specified tools and the answer is meaningful and relevant to the question.
        2. Provide a decision on whether to continue with the current plan or regenerate the plan.
        3. Provide a score (0-10) indicating how well the tools' outputs and the answer address the question.

        Example 1:
        <s>[INST]
        Question: What best describes the pool of water?
        Tools' Outputs: {{'ImageCaptionTool': 'a group of giraffes standing under a tree', 'TextDetectionTool': 'No text detected', 'KnowledgeReasoningTool': 'a pond'}}
        Multiple Choice Answer: dirty
        Direct Answer: pond
        Evaluation: Here, the Tools' Outputs provide enough information and the Answer is relevant to the question. 
        Decision: Continue
        Score: 8
        [/INST]</s>
        
        Example 2:
        <s>[INST]
        Question: Which number birthday is probably being celebrated?
        Tools' Outputs: { {'TextDetectionTool': 'No text detected', 'ImageClassificationTool': 'quill'}}
        Multiple Choice Answer: fifty
        Direct Answer: Based on the tool outputs, there is
        Evaluation: Here, None of the Tools' Outputs provide enough information and the Answer is irrelevant to the question. 
        Decision: Regenerate
        Score: 2
        [/INST]</s>
        
        Current Task:
        [INST]
        Question: {question}
        Tools' Outputs: {tool_outputs}
        Multiple Choice Answer: {mc_answer}
        Direct Answer: {da_answer}
        Evaluation: ?
        Decision: ?
        Score: ?
        [/INST]
        '''        
        inputs = self.tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id)
        decoded_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        #print("Full interaction:", decoded_text)
        evaluation_text = decoded_text.split("Score: ?")[-1].split("[/INST]")[-1].strip()
        decision = "continue" if "Continue" in evaluation_text else "regenerate"
        score = re.search(r".*(\d+)", evaluation_text).group(1) if re.search(r".*(\d+)", evaluation_text) else -1
        #print("Score:", score)
        #print(f"Evaluator Full Reply: {'>' * 30} \n {evaluation_text}")
        print(f"{'#' * 29} Evaluator Reply: {'#' * 29} \n Decision: {decision}, Score: {score}")
        return decision, score
    
class Replanner(Planner):
    def plan(self, question, image, previous_tools, previous_score, retry_count=10):
        replan_prompt = f'''
        USER: You are a re-planning agent designed to answer questions based on images by selecting the appropriate tools from a given list. Your task is to analyze the input question, image, and previous tools, determine which tools are necessary, and sequence these tools to extract the required information to answer the question.
        
        Below is a list of available tools and their functionalities:

        ImageCaptionTool - Generates captions describing the image content. Select this tool when the question requires understanding the overall scene and context of the image.
        ObjectDetectionTool - Identifies and counts objects mentioned in the query. Select this tool when the question requires verifying the presence and quantity of specific objects in the image.
        RegionGroundTool - Locates specific areas of the image based on the query. Select this tool when the question requires focusing on and analyzing relevant regions within the image.
        TextDetectionTool - Extracts text embedded within the image (OCR). Select this tool when the question requires understanding and interpreting textual information present in the image.
        TextGroundTool - Finds specific text locations in the image. Select this tool when the question requires identifying precise areas where specific text is located for further analysis.
        ImageClassificationTool - Categorizes the image into predefined classes. Select this tool when the question requires classifying objects or scenes within the image.
        ContextReasoningTool - Infers answers based on commonsense, reasoning, and the context provided by the image. Select this tool when the question requires drawing inferences, making predictions, and applying reasoning. 
        KnowledgeReasoningTool - Infers answers based on external knowledge and information retrieval. Select this tool when the question requires additional information to supplement the analysis and provide accurate answers beyond basic image understanding.
        
        Your task:
        - Select the tools required to answer the question by processing the input question and image.
        - Use the previous plan's score to guide your new selection, aiming to improve the score if it was low.
        - Your response must be in a single line, and you must only write the tool names separated by commas.
        - Your response must not contain any numbers, bullet points, or special symbols, and do not include any explanations or additional text in your answer. 
        - Your response must be in the exact format specified below. Do not deviate from this format.
        - You must not provide any explanation. Example explanations are only for your understanding.

        Example 1:
        Input: An image of a teddy bear cake on a table.
        Question: Which number birthday is probably being celebrated?
        Previous Tools: ObjectDetectionTool, TextDetectionTool 
        Previous Score: 2
        ASSISTANT: RegionGroundTool, ContextReasoningTool
        Explanation: The Previous Score indicates that the Previous Tools provided very low-quality information. Therefore, completely different tools are selected. 
        
        Example 2:
        Input: An image of a group of giraffes standing under a tree near a pond.
        Question: What best describes the pool of water?
        Previous Tools: TextDetectionTool, ImageCaptionTool 
        Previous Score: 4
        ASSISTANT: KnowledgeReasoningTool, ImageCaptionTool
        Explanation: The Previous Score indicates that the Previous Tools provided some relevant information but not enough detail. Therefore, some of the tools are changed.
        
        Current Task:
        Input: <image>
        Previous Tools: {previous_tools}
        Previous Score: {previous_score}
        Question: {question}
        ASSISTANT: ?
        '''
        inputs = self.processor(replan_prompt, image, padding=True, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.9)
        generated_text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        generated_text = generated_text.split("ASSISTANT: ?")[-1].strip()
        #print(f"Replanner Full Reply: {'<' * 30} \n {generated_text}")
        selected_tools = []
        for tool_name in self.tools:
            if tool_name in generated_text:
                selected_tools.append(tool_name)        
        if not selected_tools:
            if retry_count > 0:
                print(f"Replanner failed to select any tools. Retrying... Attempts left: {retry_count}")
                return self.plan(question, image, previous_tools, previous_score, retry_count - 1)
            else:
                print("Replanner failed to select any tools after maximum retries. Selecting all tools.")
                selected_tools = list(self.tools.keys())
        
        print(f"{'<' * 24} Selected Tools by Re-Planner: {'>' * 24} \n {selected_tools}")
        return selected_tools        
