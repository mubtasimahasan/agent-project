from model import initialize_llava, initialize_mistral 
from module import Planner, Executor, Answerer, Evaluator, Replanner
from tool import initialize_tools

#Todo: do it cleanly.
llava_processor, llava_model = initialize_llava()
mistral_model, mistral_tokenizer = initialize_mistral()
caption_tool, object_detection_tool, region_ground_tool, text_detection_tool, text_ground_tool, image_classification_tool, context_reasoning_tool, knowledge_reasoning_tool = initialize_tools()


# Define Static Pipeline
def static_pipeline(question, image, choices):
    tools = {
        "ImageCaptionTool": caption_tool,
        "ObjectDetectionTool": object_detection_tool,
        "RegionGroundTool": region_ground_tool,
        "TextDetectionTool": text_detection_tool, 
        "TextGroundTool": text_ground_tool,
        "ImageClassificationTool": image_classification_tool,
        "ContextReasoningTool": context_reasoning_tool,
        "KnowledgeReasoningTool": knowledge_reasoning_tool
    }
    
    planner = Planner(tools, llava_model, llava_processor)
    selected_tools = planner.plan(question, image)
    
    executor = Executor(tools)
    tool_results = executor.execute(question, image, selected_tools)
    
    answerer = Answerer(mistral_tokenizer, mistral_model)
    mc_answer = answerer.multiple_choice_answer(question, image, tool_results, choices)
    da_answer = answerer.direct_answer(question, image, tool_results)
    
    return tool_results, mc_answer, da_answer

# Define Dynamic Pipeline
def dynamic_pipeline(question, image, choices):
    tools = {
        "ImageCaptionTool": caption_tool,
        "ObjectDetectionTool": object_detection_tool,
        "RegionGroundTool": region_ground_tool,
        "TextDetectionTool": text_detection_tool, 
        "TextGroundTool": text_ground_tool,
        "ImageClassificationTool": image_classification_tool,
        "ContextReasoningTool": context_reasoning_tool,
        "KnowledgeReasoningTool": knowledge_reasoning_tool
    }
    
    planner = Planner(tools, llava_model, llava_processor)
    executor = Executor(tools)
    evaluator = Evaluator(mistral_tokenizer, mistral_model)
    replanner = Replanner(tools, llava_model, llava_processor)
    answerer = Answerer(mistral_tokenizer, mistral_model)

    selected_tools = planner.plan(question, image)
    tool_results = executor.execute(question, image, selected_tools)

    max_replans = 3  
    for _ in range(max_replans):
        evaluation_decision, evaluation_score = evaluator.evaluate(question, tool_results)
        if evaluation_decision == "continue":
            break
        selected_tools = replanner.plan(question, image, selected_tools, evaluation_score)
        tool_results = executor.execute(question, image, selected_tools)

    mc_answer = answerer.multiple_choice_answer(question, image, tool_results, choices)
    da_answer = answerer.direct_answer(question, image, tool_results)
    
    return tool_results, mc_answer, da_answer

# Define No-Plan Pipeline
def noplan_pipeline(question, image, choices):
    tools = {
        "ImageCaptionTool": caption_tool,
        "ObjectDetectionTool": object_detection_tool,
        "RegionGroundTool": region_ground_tool,
        "TextDetectionTool": text_detection_tool, 
        "TextGroundTool": text_ground_tool,
        "ImageClassificationTool": image_classification_tool,
        "ContextReasoningTool": context_reasoning_tool,
        "KnowledgeReasoningTool": knowledge_reasoning_tool
    }
    
    selected_tools = ["ImageCaptionTool","ObjectDetectionTool","RegionGroundTool","TextDetectionTool",
                     "TextGroundTool","ImageClassificationTool", "ContextReasoningTool", "KnowledgeReasoningTool"]
    
    print(f"Selected Tools (ALL): {'~' * 30} \n {selected_tools}")
    
    executor = Executor(tools)
    tool_results = executor.execute(question, image, selected_tools)
    
    answerer = Answerer(mistral_tokenizer, mistral_model)
    mc_answer = answerer.multiple_choice_answer(question, image, tool_results, choices)
    da_answer = answerer.direct_answer(question, image, tool_results)
    
    return tool_results, mc_answer, da_answer


