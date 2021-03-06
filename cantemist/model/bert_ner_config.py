
class Config():
    # ModelArguments
    hidden_dropout_prob = 0.5
    num_labels = 3

    # DataTrainingArguments
    data_dir = "./data_set_ner/"
    max_seq_length = 100
    output_dir = "../output/Parallel——attention"
    labels = "bio"

    # TrainingArguments
    train_batch_size = 16
    num_train_epochs = 5
    weight_decay = 0.05
    save_steps = 1300
    eval_batch_size = 16
    do_train = True
    do_predict = True
    gradient_accumulation_steps = 1
    local_rank = -1
    learning_rate = 0.0001
    warmup_steps = 0
    gpu = True
    adam_epsilon = 1e-8
    seed = 24
    max_steps = -1
    output_model_dir = "./output/model/"
    do_eval = True