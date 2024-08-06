from model import initialize_llava, initialize_mistral 
from module import Planner, Executor, Answerer, Evaluator, Replanner
from tool import initialize_tools

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
    
    # Initialize
    planner = Planner(tools, llava_model, llava_processor)
    executor = Executor(tools)
    answerer = Answerer(mistral_tokenizer, mistral_model)
    
    # Plan->Execute->Answer
    selected_tools = planner.plan(question, image)
    tool_results = executor.execute(question, image, selected_tools)
    mc_answer = answerer.multiple_choice_answer(question, tool_results, choices)
    da_answer = answerer.direct_answer(question, tool_results)
    
    return tool_results, mc_answer, da_answer


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
    
    # Initialize
    planner = Planner(tools, llava_model, llava_processor)
    executor = Executor(tools)
    answerer = Answerer(mistral_tokenizer, mistral_model)
    evaluator = Evaluator(mistral_tokenizer, mistral_model)
    replanner = Replanner(tools, llava_model, llava_processor)
    
    # Plan->Execute->Answer
    selected_tools = planner.plan(question, image)
    tool_results = executor.execute(question, image, selected_tools)
    mc_answer = answerer.multiple_choice_answer(question, tool_results, choices)
    da_answer = answerer.direct_answer(question, tool_results)
    
    max_replans = 3  
    for _ in range(max_replans):
        evaluation_decision, evaluation_score = evaluator.evaluate(
            question, tool_results, mc_answer, da_answer)
        if evaluation_decision == "continue":
            break
        
        # RePlan->Execute->Answer for mc_answer
        selected_tools = replanner.plan(question, image, selected_tools, evaluation_score)
        tool_results = executor.execute(question, image, selected_tools)
        mc_answer = answerer.multiple_choice_answer(question, tool_results, choices)
        da_answer = answerer.direct_answer(question, tool_results)
    
    # Evaluate mc_answer->
    # max_replans = 3  
    # for _ in range(max_replans):
    #     mc_evaluation_decision, mc_evaluation_score = evaluator.mc_evaluate(
    #         question, tool_results, mc_answer, da_answer)
    #     if mc_evaluation_decision == "continue":
    #         break
        
    #     # RePlan->Execute->Answer for mc_answer
    #     selected_tools = replanner.plan(question, image, selected_tools, mc_evaluation_score)
    #     tool_results = executor.execute(question, image, selected_tools)
    #     mc_answer = answerer.multiple_choice_answer(question, image, tool_results, choices)
        
    # # Evaluate da_answer->
    # for _ in range(max_replans):
    #     da_evaluation_decision, da_evaluation_score = evaluator.da_evaluate(
    #         question, tool_results, da_answer
    #     )
    #     if da_evaluation_decision == "continue":
    #         break
        
    #     # RePlan->Execute->Answer for da_answer
    #     selected_tools = replanner.plan(
    #         question, image, selected_tools, da_evaluation_score
    #     )
    #     tool_results = executor.execute(question, image, selected_tools)
    #     da_answer = answerer.direct_answer(question, image, tool_results)
        
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
    
    print(f"{'~' * 28} Selected Tools (ALL): {'~' * 28} \n {selected_tools}")
    
    executor = Executor(tools)
    tool_results = executor.execute(question, image, selected_tools)
    
    answerer = Answerer(mistral_tokenizer, mistral_model)
    mc_answer = answerer.multiple_choice_answer(question, tool_results, choices)
    da_answer = answerer.direct_answer(question, tool_results)
    
    return tool_results, mc_answer, da_answer