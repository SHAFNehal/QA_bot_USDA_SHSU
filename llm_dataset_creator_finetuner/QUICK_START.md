### An LLM was used to create this documentation.

# Quick Start: High-Quality QA Model Fine-tuning Workflow

## Step 1: Install requirements
```bash
echo "Step 1: Installing requirements..."
pip install -r requirements.txt
```

## Step 2: Generate QA dataset
```bash
echo "Step 2: Generating QA dataset..."
python dataset_creator.py \
    --input_dir data_input \
    --output_file data_output/qa_dataset.jsonl \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --chunk_size 1000 \
    --overlap 200 \
    --num_questions 3
```

## Step 3: Clean the dataset (strict quality filtering)
```bash
echo "Step 3: Cleaning dataset..."
python data_cleaner.py \
    --input data_output/qa_dataset.jsonl \
    --output data_output/qa_dataset_cleaned.jsonl \
    --verbose
```

## Step 4: Fine-tune the model
```bash
echo "Step 4: Fine-tuning model..."
python fine_tuner.py \
    --dataset_path data_output/qa_dataset_cleaned.jsonl \
    --output_dir fine_tuned_weights \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --learning_rate 3e-5 \
    --warmup_steps 100 \
    --logging_steps 20 \
    --save_steps 200
```

## Step 5: Run the fine-tuned chatbot interactively
```bash
echo "Step 5: Running fine-tuned chatbot..."
python inference.py 
    --peft_model fine_tuned_weights \
    --interactive
```

echo "=========================================="
echo "Workflow completed successfully!"
echo "=========================================="
echo "Your fine-tuned model is saved in: fine_tuned_weights"
echo "To use it interactively, run:"
echo "python inference.py --peft_model fine_tuned_weights --interactive"