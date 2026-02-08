# QA Generation Guide: Determining the Number of Question-Answer Pairs

## Overview

The number of question-answer pairs to generate per text chunk is a critical parameter that affects:
- **Dataset quality and diversity**
- **Processing time and computational cost**
- **Training effectiveness**
- **Coverage of source material**

## Factors to Consider

### 1. **Text Chunk Size**
```
Small chunks (500-1000 chars): 2-3 questions
Medium chunks (1000-2000 chars): 3-5 questions  
Large chunks (2000+ chars): 5-8 questions
```

### 2. **Content Complexity**
- **Simple factual content**: 2-3 questions (basic recall)
- **Moderate complexity**: 3-5 questions (understanding + analysis)
- **Complex technical content**: 5-8 questions (deep comprehension)

### 3. **Intended Use Case**
- **Fine-tuning for QA**: 3-5 questions per chunk
- **General instruction tuning**: 2-3 questions per chunk
- **Comprehensive coverage**: 5-8 questions per chunk

### 4. **Model Capabilities**
- **Small models (TinyLlama)**: 2-3 questions (faster, less detailed)
- **Medium models (Phi-3 Mini)**: 3-5 questions (balanced)
- **Large models (Llama 3.1)**: 5-8 questions (more comprehensive)

## Recommended Guidelines

### **Conservative Approach (Recommended for Starters)**
```python
# For most use cases
num_questions = 3  # Good balance of quality and speed
```

### **Quality-Focused Approach**
```python
# For high-quality datasets
num_questions = 5  # More comprehensive coverage
```

### **Speed-Focused Approach**
```python
# For quick testing or limited resources
num_questions = 2  # Fastest generation
```

### **Comprehensive Approach**
```python
# For maximum coverage
num_questions = 8  # Most thorough but slowest
```

## Practical Examples

### **Example 1: Academic Papers**
```python
# Complex technical content
chunk_size = 1500
num_questions = 5  # Need more questions for complex topics
```

### **Example 2: Simple Documentation**
```python
# Basic factual content
chunk_size = 800
num_questions = 3  # Fewer questions for simple content
```

### **Example 3: Research Articles**
```python
# Mixed complexity
chunk_size = 2000
num_questions = 6  # Comprehensive coverage needed
```

## Quality vs. Quantity Trade-offs

### **Higher Number of Questions (5-8)**
**Pros:**
- Better coverage of source material
- More diverse question types
- Higher chance of capturing key concepts

**Cons:**
- Slower generation
- Higher computational cost
- Potential for repetitive questions
- May overwhelm smaller models

### **Lower Number of Questions (2-3)**
**Pros:**
- Faster generation
- Lower computational cost
- Less redundancy
- Better for smaller models

**Cons:**
- May miss important concepts
- Less comprehensive coverage
- Limited question diversity

## Adaptive Strategies

### **Dynamic Question Count**
```python
def calculate_questions(chunk_size, content_complexity):
    base_questions = 3
    
    # Adjust based on chunk size
    if chunk_size > 1500:
        base_questions += 2
    elif chunk_size < 800:
        base_questions -= 1
    
    # Adjust based on complexity
    if content_complexity == "high":
        base_questions += 2
    elif content_complexity == "low":
        base_questions -= 1
    
    return max(1, min(8, base_questions))
```

### **Content-Aware Generation**
```python
# Analyze content before generating
def analyze_content(text_chunk):
    # Count technical terms
    technical_terms = count_technical_terms(text_chunk)
    
    # Count sentences
    sentence_count = len(text_chunk.split('.'))
    
    # Determine complexity
    if technical_terms > 10 or sentence_count > 8:
        return "high"
    elif technical_terms < 3 and sentence_count < 4:
        return "low"
    else:
        return "medium"
```

## Command Line Usage

### **Basic Usage**
```bash
# Default (3 questions per chunk)
python dataset_creator.py

# Conservative (2 questions per chunk)
python dataset_creator.py --num_questions 2

# Comprehensive (5 questions per chunk)
python dataset_creator.py --num_questions 5

# Maximum coverage (8 questions per chunk)
python dataset_creator.py --num_questions 8
```

### **Advanced Usage with Different Models**
```bash
# TinyLlama - conservative approach
python dataset_creator.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --num_questions 2

# Phi-3 Mini - balanced approach
python dataset_creator.py --model_name microsoft/phi-3-mini-4k-instruct --num_questions 4

# Llama 3.1 - comprehensive approach
python dataset_creator.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --num_questions 6
```

## Monitoring and Optimization

### **Quality Metrics**
- **Question diversity**: Check for repetitive questions
- **Answer quality**: Verify answers are accurate and complete
- **Coverage**: Ensure key concepts are addressed

### **Performance Metrics**
- **Generation time**: Monitor processing speed
- **Memory usage**: Track resource consumption
- **Success rate**: Count failed generations

### **Iterative Improvement**
1. **Start with 3 questions** per chunk
2. **Monitor quality** and performance
3. **Adjust based on results**:
   - If quality is poor: Increase to 4-5
   - If too slow: Decrease to 2
   - If coverage is incomplete: Increase to 5-6

## Best Practices

### **For Different Content Types**

#### **Technical Documentation**
```python
num_questions = 4  # Need to cover concepts, examples, and procedures
```

#### **Narrative Text**
```python
num_questions = 3  # Focus on main points and key details
```

#### **Research Papers**
```python
num_questions = 6  # Comprehensive coverage of methodology and findings
```

#### **Simple Instructions**
```python
num_questions = 2  # Basic procedural questions
```

### **For Different Model Sizes**

#### **Small Models (1-3B parameters)**
```python
num_questions = 2-3  # Avoid overwhelming the model
```

#### **Medium Models (3-8B parameters)**
```python
num_questions = 3-5  # Balanced approach
```

#### **Large Models (8B+ parameters)**
```python
num_questions = 5-8  # Can handle more comprehensive generation
```

## Troubleshooting

### **Common Issues**

#### **Too Many Questions**
- **Problem**: Generation fails or produces poor quality
- **Solution**: Reduce `num_questions` by 1-2

#### **Too Few Questions**
- **Problem**: Incomplete coverage of source material
- **Solution**: Increase `num_questions` by 1-2

#### **Repetitive Questions**
- **Problem**: Similar questions generated
- **Solution**: Adjust prompt or use different model

#### **Slow Generation**
- **Problem**: Takes too long to generate
- **Solution**: Reduce `num_questions` or use smaller model

## Summary

### **Recommended Starting Points**

| Use Case | Questions per Chunk | Reasoning |
|----------|-------------------|-----------|
| **Quick testing** | 2 | Fast iteration |
| **General purpose** | 3 | Good balance |
| **High quality** | 5 | Comprehensive coverage |
| **Maximum coverage** | 6-8 | Thorough but slow |

### **Key Principles**
1. **Start conservative** (2-3 questions)
2. **Monitor quality** and performance
3. **Adjust iteratively** based on results
4. **Consider content complexity** and model capabilities
5. **Balance coverage** with computational cost

### **Final Recommendation**
For most use cases, start with **3 questions per chunk** and adjust based on your specific needs and constraints.

---

## See also

- **Pipeline and config:** The main [README](../README.md) describes the full pipeline (generate → clean → merge → train with 80/10/10 train/validation/holdout split) and where to set `num_questions` (e.g. `config.py` `DEFAULT_DATASET_CONFIG`, `--questions` in `run_pipeline.sh`, or `--num_questions` in `dataset_creator.py`).
- **Evaluation:** After training, the holdout set is used for evaluation. See [evaluation.md](evaluation.md) for metrics and how to run evaluation (including on a document folder with `run_eval_on_documents.sh`). 