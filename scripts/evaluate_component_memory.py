import evaluate_token_memory_edit as evaluation
from src import component_memory_intervention


if __name__ == '__main__':
    evaluation.model_api = component_memory_intervention
    evaluation.main()
