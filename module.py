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
        #print("Full interaction with planner:", generated_text)
        generated_text = generated_text.split("ASSISTANT:")[-1].strip()
        #print("Planner Reply:", generated_text)
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
        
        print(f"Selected Tools by Planner: {'~' * 30} \n {selected_tools}")
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
        
        print(f"Tools Output: {'=' * 30} \n {results}")
        return results

class Answerer:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def multiple_choice_answer(self, question, image, results, choices):
        answer_prompt = f"""
        You are an advanced visual question answering tool.

        Task: 
        - Use previous tool outputs to determine the one-word answer to the question.
        - Your response must be exactly one word from the provided choices.
        - If the tools' outputs don't directly indicate the answer, select an answer randomly from the choices.
   
        Example:
        <s>[INST]
        Previous Tool Outputs: {{'TextDetectionTool': 'J,2 [H] 100-0237', 'ContextReasoningTool': 'frosting'}}
        Question: What is the topping on the cake?
        Choices: icing, sprinkles, candles
        Answer: icing
        [/INST]</s>

        [INST]
        Previous Tool Outputs: {results}
        Question: {question}
        Choices: {choices}
        Answer: 
        [/INST]
        """
        
        inputs = self.tokenizer(answer_prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=3, pad_token_id=self.tokenizer.eos_token_id)
        decoded_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded_text.split("Answer: ?")[-1].split("[/INST]")[-1].strip()
        return answer
    
    def direct_answer(self, question, image, results):
        answer_prompt = f"""
        You are an advanced visual question answering tool.

        Task: 
        - Use previous tool outputs to predict the one-word answer to the question.
        - Your response must be exactly one word.
        - If the tools' outputs don't directly indicate the answer, guess a probable answer to the question.

        Example:
        <s>[INST]
        Previous Tool Outputs: {{'TextDetectionTool': 'J,2 [H] 100-0237', 'ContextReasoningTool': 'frosting'}}
        Question: What is the topping on the cake?
        Answer: icing
        [/INST]</s>

        [INST]
        Previous Tool Outputs: {results}
        Question: {question}
        Answer: 
        [/INST]
        """
        
        inputs = self.tokenizer(answer_prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=3, pad_token_id=self.tokenizer.eos_token_id)
        decoded_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded_text.split("Answer: ?")[-1].split("[/INST]")[-1].strip()
        return answer
    
# Evaluator Class
class Evaluator:
    def __init__(self, tokenizer, model):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, question, tool_outputs):
        eval_prompt = f'''
        USER:
        You are an evaluation agent designed to assess the relevance and accuracy of outputs produced by various tools in answering a given question based on an image. Your task is to analyze the tools' outputs, evaluate their usefulness in the context of the question, and provide feedback to guide the planning agent. 

        Your tasks:
        1. Evaluate whether the output from the specified tools is meaningful and relevant to the question.
        2. Provide a decision on whether to continue with the current plan or regenerate the plan.
        3. Provide a score (0-10) indicating how well the tools' outputs answer the question.

        Example 1:
        <s>[INST]
        Question: What is in the motorcyclist's mouth?
        Tools' Output: {{'TextDetectionTool': 'No text detected', 'ContextReasoningTool': 'a cigar'}}
        Evaluation: Based on the ContextReasoningTool output, the correct answer is inferable, even though the TextDetectionTool did not detect any text. 
        Decision: Continue
        Score: 8
        [/INST]</s>
        
        Example 2:
        <s>[INST]
        Question: What best describes the pool of water?
        Tools' Outputs: {{'ImageCaptionTool': 'two gis are standing near a tree', 'TextDetectionTool': 'No text detected'}}
        Evaluation: None of the provided choices match the previous tool outputs or contain enough information to infer the answer from the choices.
        Decision: Regenerate
        Score: 3
        [/INST]</s>
        
        Current Task:
        [INST]
        Question: {question}
        Tool Outputs: {tool_outputs}
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
        print(f"Evaluator Reply: {'#' * 30} \n Decision: {decision}, Score: {score}")
        return decision, score
    
# Replanner Class
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
        
        print(f"Selected Tools by Replanner: {'<' * 30} \n {selected_tools}")
        return selected_tools
        
