import hydra
from omegaconf import DictConfig
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset, concatenate_datasets
import torch

CPT_CODE_PROMPT = """
# File: {file_path}

# Code Content:
{content}
"""


@hydra.main(config_path="config", config_name="cpt_train", version_base="1.3")
def main(cfg: DictConfig):

    # Load Pretrain model
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = cfg.model.name, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = cfg.model.max_seq_length,
    dtype = cfg.model.dtype,
    load_in_4bit = cfg.model.load_in_4bit,
    )

    # set peft LoRA Model
    model = FastLanguageModel.get_peft_model(
        model,
        r = cfg.lora.r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = cfg.lora.target_modules,
        lora_alpha = cfg.lora.lora_alpha,
        lora_dropout = cfg.lora.lora_dropout, # Supports any, but = 0 is optimized
        bias = cfg.lora.bias,    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = cfg.lora.use_gradient_checkpointing, # True or "unsloth" for very long context
        random_state = cfg.lora.random_state,
        use_rslora = cfg.lora.use_rslora,   # We support rank stabilized LoRA
        loftq_config = cfg.lora.loftq_config, # And LoftQ
    )

    # Data Pre Preprocessing
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        file_paths = examples["file_path"]
        contents  = examples["content"]
        outputs = []
        for file_path, content in zip(file_paths, contents):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = CPT_CODE_PROMPT.format(file_path, content) + EOS_TOKEN
            outputs.append(text)
        return {cfg.data.text_column : outputs, }
    

    # Load Multiple Data Sources
    data_sources = cfg.data.train_path
    datasets_to_combine = []
    for source in data_sources:
        if source.endswith('.json'):
            # JSON 파일 로드
            dataset = load_dataset('json', data_files=source)
            datasets_to_combine.append(dataset['train'])
        else:
            # Hugging Face 데이터셋 로드
            dataset = load_dataset(source)
            datasets_to_combine.append(dataset['train'])
    # Combine Multiple Datasets
    combined_dataset = concatenate_datasets(datasets_to_combine)
    combined_dataset = combined_dataset.map(formatting_prompts_func, batched = True,)
    
    # Check Dataset Structure
    print(f"Combined Dataset Size: {len(combined_dataset)}")
    print("First Sample:")
    print(combined_dataset[0])

    # Continued Pretraining
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = cfg.data.text_column,
        max_seq_length = cfg.model.max_seq_length,
        dataset_num_proc = 4,

        args = UnslothTrainingArguments(
            per_device_train_batch_size = cfg.training.per_device_train_batch_size,
            gradient_accumulation_steps = cfg.training.gradient_accumulation_steps,
            # max_steps = 120,
            # warmup_steps = 10,
            warmup_ratio = cfg.training.warmup_ratio,
            num_train_epochs = cfg.training.num_train_epochs,

            # Select a 2 to 10x smaller learning rate for the embedding matrices!
            learning_rate = cfg.training.learning_rate,
            embedding_learning_rate = cfg.training.embedding_learning_rate,

            logging_steps = cfg.training.logging_steps,
            optim = cfg.training.optim,
            weight_decay = cfg.training.weight_decay,
            lr_scheduler_type = cfg.training.lr_scheduler_type,
            seed = cfg.training.seed,
            output_dir = cfg.training.output_dir,
            report_to = cfg.training.report_to, # Use TrackIO/WandB etc
            
            # save strategy
            save_only_model = cfg.training.save_only_model,
            save_strategy = cfg.training.save_strategy
        ),
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    trainer.train()

    # Save Model
    trainer.save_model(cfg.save.final_model_path)


if __name__ == "__main__":
    main()
